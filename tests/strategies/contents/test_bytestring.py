from typing import TypedDict, cast

import numpy as np
from hypothesis import find, given
from hypothesis import strategies as st

import awkward as ak
from awkward.contents import ListOffsetArray, NumpyArray
from hypothesis_awkward import strategies as st_ak
from hypothesis_awkward._util import safe_compare as sc
from tests.find_settings import FIND
from tests.funcs import all_distinct, assert_kwargs_match_signature


class BytestringContentsKwargs(TypedDict, total=False):
    """Options for `bytestring_contents()` strategy."""

    min_size: int
    max_size: int
    unique: bool


DEFAULTS = BytestringContentsKwargs(min_size=0, max_size=10, unique=False)


class RelatedKwargs(TypedDict, total=False):
    """Options drawn together as an ordered pair (`min_size <= max_size`)."""

    min_size: int
    max_size: int


def test_kwargs_match_signature() -> None:
    """Assert the option declarations agree with `bytestring_contents()`."""
    assert_kwargs_match_signature(
        func=st_ak.contents.bytestring_contents,
        kwargs_cls=BytestringContentsKwargs,
        defaults=DEFAULTS,
        related_cls=RelatedKwargs,
    )


@st.composite
def bytestring_contents_kwargs(draw: st.DrawFn) -> BytestringContentsKwargs:
    """Strategy for options for `bytestring_contents()` strategy."""

    @st.composite
    def _st_related_kwargs(draw: st.DrawFn) -> RelatedKwargs:
        """Strategy for the options that are drawn together."""
        min_size, max_size = draw(
            st_ak.ranges(min_start=0, max_start=DEFAULTS['max_size'], max_end=100)
        )
        ret = RelatedKwargs()
        if min_size is not None:
            ret['min_size'] = min_size
        if max_size is not None:
            ret['max_size'] = max_size
        return ret

    related = draw(_st_related_kwargs())

    optional_independent = draw(
        st.fixed_dictionaries({}, optional={'unique': st.booleans()})
    )

    return BytestringContentsKwargs(
        **related, **cast(BytestringContentsKwargs, optional_independent)
    )


@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """Assert the results of `bytestring_contents()`."""
    # Draw options
    kwargs = data.draw(bytestring_contents_kwargs(), label='kwargs')

    # Call the test subject
    result = data.draw(st_ak.contents.bytestring_contents(**kwargs), label='result')

    # Assert the result is always a ListOffsetArray
    assert isinstance(result, ListOffsetArray)

    # Assert outer __array__ parameter
    assert result.parameter('__array__') == 'bytestring'

    # Assert inner content is NumpyArray
    assert isinstance(result.content, NumpyArray)

    # Assert inner __array__ parameter
    assert result.content.parameter('__array__') == 'byte'

    # Assert inner dtype is uint8
    assert result.content.dtype == np.dtype(np.uint8)

    # Assert size bounds
    min_size = kwargs.get('min_size', DEFAULTS['min_size'])
    max_size = kwargs.get('max_size', DEFAULTS['max_size'])
    unique = kwargs.get('unique', DEFAULTS['unique'])

    assert sc(min_size) <= len(result) <= sc(max_size)

    # Assert all elements are bytes
    arr = ak.Array(result)
    for b in arr.to_list():
        assert isinstance(b, bytes)

    if unique:
        assert all_distinct(arr.to_list())


def test_draw_empty() -> None:
    """Assert that empty arrays (zero bytestrings) can be drawn."""
    find(st_ak.contents.bytestring_contents(), lambda c: len(c) == 0, settings=FIND)


def test_draw_max_size() -> None:
    """Assert that arrays with exactly max_size bytestrings can be drawn."""
    find(
        st_ak.contents.bytestring_contents(),
        lambda c: len(c) == DEFAULTS['max_size'],
        settings=FIND,
    )


def test_draw_unique_max_size() -> None:
    """Assert that unique arrays with exactly max_size bytestrings can be drawn."""
    find(
        st_ak.contents.bytestring_contents(unique=True),
        lambda c: len(c) == DEFAULTS['max_size'],
        settings=FIND,
    )


def test_draw_empty_bytestring() -> None:
    """Assert that arrays containing an empty bytestring can be drawn."""
    find(
        st_ak.contents.bytestring_contents(min_size=1),
        lambda c: b'' in ak.Array(c).to_list(),
        settings=FIND,
    )


def test_draw_null_byte() -> None:
    """Assert that arrays containing a bytestring with a null byte can be drawn."""
    find(
        st_ak.contents.bytestring_contents(min_size=1),
        lambda c: any(b'\x00' in b for b in ak.Array(c).to_list()),
        settings=FIND,
    )
