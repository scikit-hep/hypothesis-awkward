from typing import Any, TypedDict, cast

import numpy as np
import pytest
from hypothesis import Phase, find, given, settings
from hypothesis import strategies as st

import awkward as ak
from awkward.contents import Content, IndexedArray
from hypothesis_awkward import strategies as st_ak
from hypothesis_awkward.util import iter_contents
from hypothesis_awkward.util import safe_compare as sc


class IndexedArrayContentsKwargs(TypedDict, total=False):
    """Options for `indexed_array_contents()` strategy."""

    content: st.SearchStrategy[Content] | Content
    min_size: int
    max_size: int


@st.composite
def indexed_array_contents_kwargs(
    draw: st.DrawFn,
    chain: st_ak.OptsChain[Any] | None = None,
) -> st_ak.OptsChain[IndexedArrayContentsKwargs]:
    """Strategy for options for `indexed_array_contents()` strategy."""
    if chain is None:
        chain = st_ak.OptsChain({})
    st_content = chain.register(
        st_ak.contents.contents(
            allow_union_root=False,
            allow_option_root=False,
            allow_indexed_root=False,
        )
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
                        allow_union_root=False,
                        allow_option_root=False,
                        allow_indexed_root=False,
                    ),
                    st.just(st_content),
                ),
            },
        )
    )

    return chain.extend(cast(IndexedArrayContentsKwargs, kwargs))


@settings(max_examples=200)
@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """Assert the results of `indexed_array_contents()`."""
    # Draw options
    opts = data.draw(indexed_array_contents_kwargs(), label='opts')
    opts.reset()

    # Call the test subject
    result = data.draw(
        st_ak.contents.indexed_array_contents(**opts.kwargs), label='result'
    )

    assert isinstance(result, IndexedArray)

    # Assert index length is within bounds
    min_size = opts.kwargs.get('min_size', 0)
    max_size = opts.kwargs.get('max_size')
    assert min_size <= len(result) <= sc(max_size)

    # Assert index dtype is int32, uint32, or int64
    assert result.index.data.dtype in (np.int32, np.uint32, np.int64)

    # Assert all index values are valid content indices (no missing entries)
    index = result.index.data
    content_len = len(result.content)
    for i in range(len(index)):
        assert 0 <= index[i] < content_len

    # Assert type transparency: an IndexedArray has "exactly the same type as
    # its content" (it only rearranges elements). Compare element types via
    # `form.type`; the array type would also encode length, which can differ.
    assert result.form.type == result.content.form.type
    # The content must not be an option, indexed, or union node (the constructor
    # forbids those children).
    assert not result.content.is_option
    assert not result.content.is_indexed
    assert not result.content.is_union

    # Assert the layout is valid by round-tripping through a high-level array
    arr = ak.Array(result)
    assert len(arr) == len(result)

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
        st_ak.contents.indexed_array_contents(min_size=min_size),
        lambda c: len(c) == min_size,
    )


@pytest.mark.parametrize('dtype', [np.int32, np.uint32, np.int64])
def test_draw_index_dtype(dtype: np.dtype) -> None:
    """Assert the given index dtype can be drawn."""
    find(
        st_ak.contents.indexed_array_contents(),
        lambda c: c.index.data.dtype == dtype,
    )


def test_draw_duplicate_indices() -> None:
    """Assert that duplicate index entries can appear."""
    find(
        st_ak.contents.indexed_array_contents(),
        lambda c: len(c) >= 2 and len(set(c.index.data.tolist())) < len(c),
    )


def test_draw_index_longer_than_content() -> None:
    """Assert that the index can be longer than the content."""
    find(
        st_ak.contents.indexed_array_contents(),
        lambda c: len(c) > len(c.content),
    )


def test_draw_permutation() -> None:
    """Assert a pure rearrangement (no missing, no duplicates) can be drawn."""
    find(
        st_ak.contents.indexed_array_contents(),
        lambda c: (
            len(c) == len(c.content) >= 2
            and sorted(c.index.data.tolist()) == list(range(len(c.content)))
        ),
    )


def test_draw_empty_content() -> None:
    """Assert that empty content forces a zero-length result."""
    find(
        st_ak.contents.indexed_array_contents(),
        lambda c: len(c.content) == 0 and len(c) == 0,
        settings=settings(max_examples=2000),
    )


def test_draw_from_contents() -> None:
    """Assert `contents()` can generate an `IndexedArray` as outermost."""
    find(
        st_ak.contents.contents(),
        lambda c: isinstance(c, IndexedArray),
        settings=settings(max_examples=2000),
    )


def test_draw_nested_indexed() -> None:
    """Assert an IndexedArray with a descendant IndexedArray can be drawn.

    The direct subcontent of an IndexedArray must not be an IndexedArray
    (no `Indexed[Indexed[...]]`), but deeper descendants may be, e.g.
    `Indexed[Regular[Indexed[...]]]`.
    """
    find(
        st_ak.contents.indexed_array_contents(),
        _has_nested_indexed,
        settings=settings(phases=[Phase.generate], max_examples=5000),
    )


def _has_nested_indexed(c: Content) -> bool:
    for node in iter_contents(c):
        if not isinstance(node, IndexedArray):
            continue
        # A direct subcontent of an IndexedArray cannot be an IndexedArray; a
        # deeper descendant may be, e.g. Indexed[Regular[Indexed[...]]].
        for descendant in iter_contents(node.content):
            if isinstance(descendant, IndexedArray):
                return True
    return False
