from typing import Any, TypedDict, cast

import numpy as np
import pytest
from hypothesis import find, given, settings
from hypothesis import strategies as st

from awkward.contents import Content, IndexedOptionArray
from hypothesis_awkward import strategies as st_ak
from hypothesis_awkward.util import safe_compare as sc


class IndexedOptionArrayContentsKwargs(TypedDict, total=False):
    """Options for `indexed_option_array_contents()` strategy."""

    content: st.SearchStrategy[Content] | Content
    min_size: int
    max_size: int


@st.composite
def indexed_option_array_contents_kwargs(
    draw: st.DrawFn,
    chain: st_ak.OptsChain[Any] | None = None,
) -> st_ak.OptsChain[IndexedOptionArrayContentsKwargs]:
    """Strategy for options for `indexed_option_array_contents()` strategy."""
    if chain is None:
        chain = st_ak.OptsChain({})
    st_content = chain.register(
        st_ak.contents.contents(allow_union_root=False, allow_option_root=False)
    )

    min_size, max_size = draw(st_ak.ranges(min_start=0, max_end=50))

    drawn = (
        ('min_size', min_size),
        ('max_size', max_size),
    )

    kwargs = draw(
        st.fixed_dictionaries(
            {k: st.just(v) for k, v in drawn if v is not None},
            optional={
                'content': st.one_of(
                    st_ak.contents.contents(
                        allow_union_root=False, allow_option_root=False
                    ),
                    st.just(st_content),
                ),
            },
        )
    )

    return chain.extend(cast(IndexedOptionArrayContentsKwargs, kwargs))


@settings(max_examples=200)
@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """Assert the results of `indexed_option_array_contents()`."""
    # Draw options
    opts = data.draw(indexed_option_array_contents_kwargs(), label='opts')
    opts.reset()

    # Call the test subject
    result = data.draw(
        st_ak.contents.indexed_option_array_contents(**opts.kwargs), label='result'
    )

    assert isinstance(result, IndexedOptionArray)

    # Assert index length is within bounds
    min_size = opts.kwargs.get('min_size', 0)
    max_size = opts.kwargs.get('max_size')
    assert min_size <= len(result) <= sc(max_size)

    # Assert index dtype is int32 or int64
    assert result.index.data.dtype in (np.int32, np.int64)

    # Assert all index values are -1 or valid content indices
    index = result.index.data
    content_len = len(result.content)
    for i in range(len(index)):
        assert index[i] == -1 or (0 <= index[i] < content_len)

    # Assert the options were effective
    content = opts.kwargs.get('content', None)
    match content:
        case Content():
            assert result.content is content
        case st_ak.RecordDraws():
            assert len(content.drawn) == 1
            assert result.content is content.drawn[0]


@pytest.mark.parametrize('min_size', [1, 2, 10])
def test_draw_min_size(min_size: int) -> None:
    """Assert the index length can reach `min_size`."""
    find(
        st_ak.contents.indexed_option_array_contents(min_size=min_size),
        lambda c: len(c) == min_size,
    )


@pytest.mark.parametrize('dtype', [np.int32, np.int64])
def test_draw_index_dtype(dtype: np.dtype) -> None:
    """Assert the given index dtype can be drawn."""
    find(
        st_ak.contents.indexed_option_array_contents(),
        lambda c: c.index.data.dtype == dtype,
    )


def test_draw_with_none_values() -> None:
    """Assert that missing (-1) entries can appear."""
    find(
        st_ak.contents.indexed_option_array_contents(),
        lambda c: len(c) > 0 and any(c.index.data[i] < 0 for i in range(len(c))),
    )


def test_draw_all_valid() -> None:
    """Assert that all-valid arrays (no missing entries) can be drawn."""
    find(
        st_ak.contents.indexed_option_array_contents(),
        lambda c: len(c) > 0 and all(c.index.data[i] >= 0 for i in range(len(c))),
    )


def test_draw_duplicate_indices() -> None:
    """Assert that duplicate index entries can appear."""
    find(
        st_ak.contents.indexed_option_array_contents(),
        lambda c: (
            len(c) >= 2
            and len(set(i for i in c.index.data if i >= 0))
            < sum(1 for i in c.index.data if i >= 0)
        ),
    )


def test_draw_index_longer_than_content() -> None:
    """Assert that the index can be longer than the content."""
    find(
        st_ak.contents.indexed_option_array_contents(),
        lambda c: len(c) > len(c.content),
    )


def test_draw_from_contents() -> None:
    """Assert `contents()` can generate an `IndexedOptionArray` as outermost."""
    find(
        st_ak.contents.contents(),
        lambda c: isinstance(c, IndexedOptionArray),
        settings=settings(max_examples=2000),
    )
