from contextlib import ExitStack
from typing import Any, TypedDict, cast

import numpy as np
import pytest
from hypothesis import find, given, settings
from hypothesis import strategies as st

from awkward.contents import EmptyArray, NumpyArray
from hypothesis_awkward import strategies as st_ak
from hypothesis_awkward.util import (
    any_nan_in_awkward_array,
    any_nan_nat_in_numpy_array,
    is_bytestring_leaf,
    is_string_leaf,
)
from hypothesis_awkward.util import safe_compare as sc

DEFAULT_MAX_SIZE = 10


class LeafContentsKwargs(TypedDict, total=False):
    """Options for `leaf_contents()` strategy."""

    dtypes: st.SearchStrategy[np.dtype] | None
    allow_nan: bool
    min_size: int
    max_size: int
    allow_numpy: bool
    allow_empty: bool
    allow_string: bool
    allow_bytestring: bool


@st.composite
def leaf_contents_kwargs(
    draw: st.DrawFn,
    chain: st_ak.OptsChain[Any] | None = None,
) -> st_ak.OptsChain[LeafContentsKwargs]:
    """Strategy for options for `leaf_contents()` strategy."""
    if chain is None:
        chain = st_ak.OptsChain({})
    st_dtypes = chain.register(st_ak.supported_dtypes())

    min_size, max_size = draw(
        st_ak.ranges(
            min_start=0, max_start=DEFAULT_MAX_SIZE, max_end=DEFAULT_MAX_SIZE + 50
        )
    )

    drawn = (
        ('min_size', min_size),
        ('max_size', max_size),
    )

    kwargs = draw(
        st.fixed_dictionaries(
            {k: st.just(v) for k, v in drawn if v is not None},
            optional={
                'dtypes': st.one_of(
                    st.none(),
                    st.just(st_dtypes),
                ),
                'allow_nan': st.booleans(),
                'allow_numpy': st.booleans(),
                'allow_empty': st.booleans(),
                'allow_string': st.booleans(),
                'allow_bytestring': st.booleans(),
            },
        )
    )

    return chain.extend(cast(LeafContentsKwargs, kwargs))


@settings(max_examples=200)
@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """Assert the results of `leaf_contents()`."""
    # Draw options
    opts = data.draw(leaf_contents_kwargs(), label='opts')
    opts.reset()

    min_size = opts.kwargs.get('min_size', 0)
    max_size = opts.kwargs.get('max_size', DEFAULT_MAX_SIZE)
    allow_numpy = opts.kwargs.get('allow_numpy', True)
    allow_empty = opts.kwargs.get('allow_empty', True)
    allow_string = opts.kwargs.get('allow_string', True)
    allow_bytestring = opts.kwargs.get('allow_bytestring', True)

    def _is_allowed_any() -> bool:
        return any(
            (
                min_size <= 0 <= max_size and allow_empty,
                allow_numpy,
                allow_string,
                allow_bytestring,
            )
        )

    # Call the test subject
    expect_raised = False
    with ExitStack() as stack:
        if not _is_allowed_any():
            expect_raised = True
            stack.enter_context(pytest.raises(ValueError))
        result = data.draw(st_ak.contents.leaf_contents(**opts.kwargs), label='result')

    if expect_raised:
        return

    is_numpy = isinstance(result, NumpyArray)
    is_empty = isinstance(result, EmptyArray)
    is_string = is_string_leaf(result)
    is_bytestring = is_bytestring_leaf(result)

    assert any((is_numpy, is_empty, is_string, is_bytestring))

    # Assert the options were effective
    if not allow_numpy:
        assert not is_numpy
    if not allow_empty:
        assert not is_empty
    if not allow_string:
        assert not is_string
    if not allow_bytestring:
        assert not is_bytestring

    dtypes = opts.kwargs.get('dtypes', None)
    allow_nan = opts.kwargs.get('allow_nan', True)

    if is_empty:
        assert len(result) == 0
    else:
        assert sc(min_size) <= len(result) <= sc(max_size)

    if is_numpy:
        assert isinstance(result, NumpyArray)
        if not allow_nan:
            assert not any_nan_nat_in_numpy_array(result.data)
        match dtypes:
            case st_ak.RecordDraws():
                drawn_kinds = {d.kind for d in dtypes.drawn}
                assert result.data.dtype.kind in drawn_kinds


def test_draw_numpy_array() -> None:
    """Assert that NumpyArray can be drawn by default."""
    find(st_ak.contents.leaf_contents(), lambda c: isinstance(c, NumpyArray))


def test_draw_empty_array() -> None:
    """Assert that EmptyArray can be drawn by default."""
    find(st_ak.contents.leaf_contents(), lambda c: isinstance(c, EmptyArray))


def test_draw_string() -> None:
    """Assert that string content can be drawn by default."""
    find(st_ak.contents.leaf_contents(), lambda c: is_string_leaf(c))


def test_draw_bytestring() -> None:
    """Assert that bytestring content can be drawn by default."""
    find(st_ak.contents.leaf_contents(), lambda c: is_bytestring_leaf(c))


def test_draw_max_size() -> None:
    """Assert that leaf content with max_size elements can be drawn."""
    max_size = 25
    find(
        st_ak.contents.leaf_contents(max_size=max_size),
        lambda c: len(c) == max_size,
        settings=settings(max_examples=2000),
    )


def test_draw_nan() -> None:
    """Assert that leaf content with NaN can be drawn."""
    c = find(
        st_ak.contents.leaf_contents(),
        any_nan_in_awkward_array,
        settings=settings(max_examples=10_000, database=None),
    )
    assert isinstance(c, NumpyArray)
    assert np.array_equal(c.data, np.array([np.nan]), equal_nan=True)


def test_shrink_leaf_contents_empty() -> None:
    """Assert that leaf content can shrink to an empty array when allowed."""
    c = find(
        st_ak.contents.leaf_contents(),
        lambda c: len(c) == 0,
        settings=settings(max_examples=2000, database=None),
    )
    assert isinstance(c, EmptyArray)


def test_shrink_leaf_contents_one() -> None:
    """Assert that leaf content with one element shrinks to [b'']."""
    c = find(
        st_ak.contents.leaf_contents(),
        lambda c: len(c) == 1,
        settings=settings(database=None),
    )
    assert c.to_list() == [b'']
