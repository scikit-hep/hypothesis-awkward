from typing import TypedDict

import numpy as np
import pytest
from hypothesis import find, given
from hypothesis import strategies as st

import awkward as ak
from awkward.contents import Content, IndexedArray
from hypothesis_awkward import strategies as st_ak
from hypothesis_awkward._util import iter_contents
from hypothesis_awkward._util import safe_compare as sc
from tests.find_settings import FIND, FIND_RARE, FIND_RARE_NO_SHRINK
from tests.funcs import KwargsDrawn, assert_kwargs_match_signature


class IndexedArrayContentsKwargs(TypedDict, total=False):
    """Options for `indexed_array_contents()` strategy."""

    content: st.SearchStrategy[Content] | Content | None
    min_size: int
    max_size: int | None


DEFAULTS = IndexedArrayContentsKwargs(
    content=None,
    min_size=0,
    max_size=None,
)


class IndexedArrayContentsDrawn(TypedDict):
    """Values drawn in advance for the options of `indexed_array_contents()`."""

    content: Content | None


class RelatedKwargs(TypedDict, total=False):
    """Options drawn together as an ordered pair (`min_size <= max_size`)."""

    min_size: int
    max_size: int | None


def test_kwargs_match_signature() -> None:
    """Assert the option declarations agree with `indexed_array_contents()`."""
    assert_kwargs_match_signature(
        func=st_ak.contents.indexed_array_contents,
        kwargs_cls=IndexedArrayContentsKwargs,
        defaults=DEFAULTS,
        related_cls=RelatedKwargs,
    )


@st.composite
def indexed_array_contents_kwargs(
    draw: st.DrawFn,
) -> KwargsDrawn[IndexedArrayContentsKwargs, IndexedArrayContentsDrawn]:
    """Strategy for options for `indexed_array_contents()` strategy."""

    @st.composite
    def _st_related_kwargs(draw: st.DrawFn) -> RelatedKwargs:
        """Strategy for the options that are drawn together."""
        min_size, max_size = draw(st_ak.ranges(min_start=0, max_end=50))
        ret = RelatedKwargs()
        if min_size is not None:
            ret['min_size'] = min_size
        if max_size is not None:
            ret['max_size'] = max_size
        return ret

    related = draw(_st_related_kwargs())

    optional_independent = IndexedArrayContentsKwargs()
    drawn = IndexedArrayContentsDrawn(content=None)
    if draw(st.booleans()):
        content = draw(
            st_ak.contents.contents(
                allow_union_root=False,
                allow_option_root=False,
                allow_indexed_root=False,
            )
        )
        optional_independent['content'] = draw(
            st.sampled_from([content, st.just(content)])
        )
        drawn['content'] = content

    kwargs = IndexedArrayContentsKwargs(**related, **optional_independent)
    return KwargsDrawn(kwargs, drawn)


@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """Assert the results of `indexed_array_contents()`."""
    # Draw options
    kwargs, drawn = data.draw(indexed_array_contents_kwargs(), label='opts')

    # Call the test subject
    result = data.draw(st_ak.contents.indexed_array_contents(**kwargs), label='result')

    assert isinstance(result, IndexedArray)

    # Assert index length is within bounds
    min_size = kwargs.get('min_size', DEFAULTS['min_size'])
    max_size = kwargs.get('max_size', DEFAULTS['max_size'])
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
    if 'content' in kwargs:
        assert result.content is drawn['content']


@pytest.mark.parametrize('min_size', [1, 2, 10])
def test_draw_min_size(min_size: int) -> None:
    """Assert the index length can reach `min_size`."""
    find(
        st_ak.contents.indexed_array_contents(min_size=min_size),
        lambda c: len(c) == min_size,
        settings=FIND,
    )


@pytest.mark.parametrize('dtype', [np.int32, np.uint32, np.int64])
def test_draw_index_dtype(dtype: np.dtype) -> None:
    """Assert the given index dtype can be drawn."""
    find(
        st_ak.contents.indexed_array_contents(),
        lambda c: c.index.data.dtype == dtype,
        settings=FIND,
    )


def test_draw_duplicate_indices() -> None:
    """Assert that duplicate index entries can appear."""
    find(
        st_ak.contents.indexed_array_contents(),
        lambda c: len(c) >= 2 and len(set(c.index.data.tolist())) < len(c),
        settings=FIND,
    )


def test_draw_index_longer_than_content() -> None:
    """Assert that the index can be longer than the content."""
    find(
        st_ak.contents.indexed_array_contents(),
        lambda c: len(c) > len(c.content),
        settings=FIND,
    )


def test_draw_permutation() -> None:
    """Assert a pure rearrangement (no missing, no duplicates) can be drawn."""
    find(
        st_ak.contents.indexed_array_contents(),
        lambda c: (
            len(c) == len(c.content) >= 2
            and sorted(c.index.data.tolist()) == list(range(len(c.content)))
        ),
        settings=FIND_RARE,
    )


def test_draw_empty_content() -> None:
    """Assert that empty content forces a zero-length result."""
    find(
        st_ak.contents.indexed_array_contents(),
        lambda c: len(c.content) == 0 and len(c) == 0,
        settings=FIND,
    )


def test_draw_from_contents() -> None:
    """Assert `contents()` can generate an `IndexedArray` as outermost."""
    find(
        st_ak.contents.contents(), lambda c: isinstance(c, IndexedArray), settings=FIND
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
        settings=FIND_RARE_NO_SHRINK,
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
