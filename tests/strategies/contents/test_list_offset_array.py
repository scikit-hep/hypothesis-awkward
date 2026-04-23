from typing import Any, TypedDict, cast

import numpy as np
import pytest
from hypothesis import find, given, settings
from hypothesis import strategies as st

from awkward.contents import Content, ListOffsetArray, NumpyArray
from hypothesis_awkward import strategies as st_ak
from hypothesis_awkward.util import safe_compare as sc


class ListOffsetArrayContentsKwargs(TypedDict, total=False):
    """Options for `list_offset_array_contents()` strategy."""

    content: st.SearchStrategy[Content] | Content
    max_length: int | None


@st.composite
def list_offset_array_contents_kwargs(
    draw: st.DrawFn,
    chain: st_ak.OptsChain[Any] | None = None,
) -> st_ak.OptsChain[ListOffsetArrayContentsKwargs]:
    """Strategy for options for `list_offset_array_contents()` strategy."""
    if chain is None:
        chain = st_ak.OptsChain({})
    st_content = chain.register(st_ak.contents.contents())

    kwargs = draw(
        st.fixed_dictionaries(
            {},
            optional={
                'content': st.one_of(
                    st_ak.contents.contents(),
                    st.just(st_content),
                ),
                'max_length': st.integers(min_value=0, max_value=50),
            },
        )
    )

    return chain.extend(cast(ListOffsetArrayContentsKwargs, kwargs))


@settings(max_examples=200)
@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """Assert the results of `list_offset_array_contents()`."""
    # Draw options
    opts = data.draw(list_offset_array_contents_kwargs(), label='opts')
    opts.reset()

    # Call the test subject
    result = data.draw(
        st_ak.contents.list_offset_array_contents(**opts.kwargs), label='result'
    )

    assert isinstance(result, ListOffsetArray)

    # Assert the options were effective
    max_length = opts.kwargs.get('max_length')
    assert len(result) <= sc(max_length)

    # Assert offsets are monotonically non-decreasing
    offsets = result.offsets.data
    for i in range(len(offsets) - 1):
        assert offsets[i] <= offsets[i + 1]

    # Assert offsets are within content bounds
    # TODO: Re-enable when allow_unreachable option is added to
    # list_offset_array_contents().
    # assert offsets[0] == 0
    # assert offsets[-1] == len(result.content)
    assert offsets[0] >= 0
    assert offsets[-1] <= len(result.content)

    # Assert the options were effective
    content = opts.kwargs.get('content', None)
    match content:
        case Content():
            assert result.content is content
        case st_ak.RecordDraws():
            assert len(content.drawn) == 1
            assert result.content is content.drawn[0]


def test_draw_max_length() -> None:
    """Assert that max_length constrains the ListOffsetArray length."""
    max_length = 10
    find(
        st_ak.contents.list_offset_array_contents(max_length=max_length),
        lambda c: len(c) == max_length,
    )


def test_draw_default_max_length() -> None:
    """Assert that len(result) can reach len(content) by default."""
    content = NumpyArray(np.zeros(20, dtype=np.int64))
    find(
        st_ak.contents.list_offset_array_contents(content),
        lambda c: len(c) == len(content),
    )


def test_draw_variable_length() -> None:
    """Assert that variable-length sublists can be drawn."""
    find(
        st_ak.contents.list_offset_array_contents(),
        lambda c: len(c) >= 2 and len(set(len(c[i]) for i in range(len(c)))) > 1,
    )


def test_draw_empty_sublist() -> None:
    """Assert that empty sublists can be drawn."""
    find(
        st_ak.contents.list_offset_array_contents(),
        lambda c: any(len(c[i]) == 0 for i in range(len(c))),
    )


def test_draw_unreachable() -> None:
    """Assert that ListOffsetArray with unreachable data can be drawn."""
    content = NumpyArray(np.arange(10))
    find(
        st_ak.contents.list_offset_array_contents(content),
        lambda c: c.offsets[0] > 0 or c.offsets[-1] < len(c.content),
    )


@pytest.mark.xfail(reason='shrinker does not reliably reach no-unreachable layout')
def test_shrink_no_unreachable() -> None:
    """Assert that ListOffsetArray shrinks to no unreachable data."""
    content = NumpyArray(np.arange(10))
    c = find(
        st_ak.contents.list_offset_array_contents(content),
        lambda c: len(c) >= 2,
    )
    assert c.offsets[0] == 0
    assert c.offsets[-1] == len(c.content)


def test_shrink_content_len_zero() -> None:
    """Assert that ListOffsetArray shrinks to zero lists with no content."""
    content = NumpyArray(np.array([], dtype=np.int64))
    c = find(
        st_ak.contents.list_offset_array_contents(content),
        lambda c: True,
    )
    assert len(c) == 0


def test_draw_from_contents() -> None:
    """Assert `contents()` can generate a `ListOffsetArray` as outermost."""
    find(st_ak.contents.contents(), lambda c: isinstance(c, ListOffsetArray))
