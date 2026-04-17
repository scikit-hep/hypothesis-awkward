from typing import Any, TypedDict, cast

import numpy as np
from hypothesis import Phase, find, given, settings
from hypothesis import strategies as st

from awkward.contents import Content, IndexedOptionArray
from hypothesis_awkward import strategies as st_ak
from hypothesis_awkward.util import safe_compare as sc


class IndexedOptionArrayContentsKwargs(TypedDict, total=False):
    """Options for `indexed_option_array_contents()` strategy."""

    content: st.SearchStrategy[Content] | Content
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

    kwargs = draw(
        st.fixed_dictionaries(
            {},
            optional={
                'content': st.one_of(
                    st_ak.contents.contents(
                        allow_union_root=False, allow_option_root=False
                    ),
                    st.just(st_content),
                ),
                'max_size': st.integers(min_value=0, max_value=50),
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

    # Assert max_size constrains index length
    max_size = opts.kwargs.get('max_size')
    assert len(result) <= sc(max_size)

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


def test_draw_index_dtype_int32() -> None:
    """Assert that int32 index dtype can be drawn."""
    find(
        st_ak.contents.indexed_option_array_contents(),
        lambda c: c.index.data.dtype == np.int32,
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_index_dtype_int64() -> None:
    """Assert that int64 index dtype can be drawn."""
    find(
        st_ak.contents.indexed_option_array_contents(),
        lambda c: c.index.data.dtype == np.int64,
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_from_contents() -> None:
    """Assert `contents()` can generate an `IndexedOptionArray` as outermost."""
    find(
        st_ak.contents.contents(),
        lambda c: isinstance(c, IndexedOptionArray),
        settings=settings(max_examples=2000),
    )


def test_draw_with_none_values() -> None:
    """Assert that missing (-1) entries can appear."""
    find(
        st_ak.contents.indexed_option_array_contents(),
        lambda c: len(c) > 0 and any(c.index.data[i] < 0 for i in range(len(c))),
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_all_valid() -> None:
    """Assert that all-valid arrays (no missing entries) can be drawn."""
    find(
        st_ak.contents.indexed_option_array_contents(),
        lambda c: len(c) > 0 and all(c.index.data[i] >= 0 for i in range(len(c))),
        settings=settings(phases=[Phase.generate], max_examples=2000),
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
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_index_longer_than_content() -> None:
    """Assert that the index can be longer than the content."""
    find(
        st_ak.contents.indexed_option_array_contents(),
        lambda c: len(c) > len(c.content),
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )
