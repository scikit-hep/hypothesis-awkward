from typing import TypedDict

import numpy as np
import pytest
from hypothesis import find, given
from hypothesis import strategies as st

from awkward.contents import Content, IndexedOptionArray
from hypothesis_awkward import strategies as st_ak
from hypothesis_awkward._util import safe_compare as sc
from tests.find_settings import FIND
from tests.funcs import KwargsDrawn, assert_kwargs_match_signature


class IndexedOptionArrayContentsKwargs(TypedDict, total=False):
    """Options for `indexed_option_array_contents()` strategy."""

    content: st.SearchStrategy[Content] | Content | None
    min_size: int
    max_size: int | None


DEFAULTS = IndexedOptionArrayContentsKwargs(
    content=None,
    min_size=0,
    max_size=None,
)


class IndexedOptionArrayContentsDrawn(TypedDict):
    """Values drawn in advance for the options of `indexed_option_array_contents()`."""

    content: Content | None


class RelatedKwargs(TypedDict, total=False):
    """Options drawn together as an ordered pair (`min_size <= max_size`)."""

    min_size: int
    max_size: int | None


def test_kwargs_match_signature() -> None:
    """Assert the option declarations agree with `indexed_option_array_contents()`."""
    assert_kwargs_match_signature(
        func=st_ak.indexed_option_array_contents,
        kwargs_cls=IndexedOptionArrayContentsKwargs,
        defaults=DEFAULTS,
        related_cls=RelatedKwargs,
    )


@st.composite
def indexed_option_array_contents_kwargs(
    draw: st.DrawFn,
) -> KwargsDrawn[IndexedOptionArrayContentsKwargs, IndexedOptionArrayContentsDrawn]:
    """Strategy for options for `indexed_option_array_contents()` strategy."""

    @st.composite
    def _st_related_kwargs(draw: st.DrawFn) -> RelatedKwargs:
        """Strategy for the options that are drawn together."""
        min_size, max_size = draw(st_ak.misc.ranges(min_start=0, max_end=50))
        ret = RelatedKwargs()
        if min_size is not None:
            ret['min_size'] = min_size
        if max_size is not None:
            ret['max_size'] = max_size
        return ret

    related = draw(_st_related_kwargs())

    optional_independent = IndexedOptionArrayContentsKwargs()
    drawn = IndexedOptionArrayContentsDrawn(content=None)
    if draw(st.booleans()):
        content = draw(
            st_ak.contents(
                allow_union_root=False,
                allow_option_root=False,
                allow_indexed_root=False,
            )
        )
        optional_independent['content'] = draw(
            st.sampled_from([content, st.just(content)])
        )
        drawn['content'] = content

    kwargs = IndexedOptionArrayContentsKwargs(**related, **optional_independent)
    return KwargsDrawn(kwargs, drawn)


@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """Assert the results of `indexed_option_array_contents()`."""
    # Draw options
    kwargs, drawn = data.draw(indexed_option_array_contents_kwargs(), label='opts')

    # Call the test subject
    result = data.draw(st_ak.indexed_option_array_contents(**kwargs), label='result')

    assert isinstance(result, IndexedOptionArray)

    # Assert index length is within bounds
    min_size = kwargs.get('min_size', DEFAULTS['min_size'])
    max_size = kwargs.get('max_size', DEFAULTS['max_size'])
    assert min_size <= len(result) <= sc(max_size)

    # Assert index dtype is int32 or int64
    assert result.index.data.dtype in (np.int32, np.int64)

    # Assert all index values are -1 or valid content indices
    index = result.index.data
    content_len = len(result.content)
    for i in range(len(index)):
        assert index[i] == -1 or (0 <= index[i] < content_len)

    # Assert the options were effective
    if 'content' in kwargs:
        assert result.content is drawn['content']


@pytest.mark.parametrize('min_size', [1, 2, 10])
def test_draw_min_size(min_size: int) -> None:
    """Assert the index length can reach `min_size`."""
    find(
        st_ak.indexed_option_array_contents(min_size=min_size),
        lambda c: len(c) == min_size,
        settings=FIND,
    )


@pytest.mark.parametrize('dtype', [np.int32, np.int64])
def test_draw_index_dtype(dtype: np.dtype) -> None:
    """Assert the given index dtype can be drawn."""
    find(
        st_ak.indexed_option_array_contents(),
        lambda c: c.index.data.dtype == dtype,
        settings=FIND,
    )


def test_draw_with_none_values() -> None:
    """Assert that missing (-1) entries can appear."""
    find(
        st_ak.indexed_option_array_contents(),
        lambda c: len(c) > 0 and any(c.index.data[i] < 0 for i in range(len(c))),
        settings=FIND,
    )


def test_draw_all_valid() -> None:
    """Assert that all-valid arrays (no missing entries) can be drawn."""
    find(
        st_ak.indexed_option_array_contents(),
        lambda c: len(c) > 0 and all(c.index.data[i] >= 0 for i in range(len(c))),
        settings=FIND,
    )


def test_draw_duplicate_indices() -> None:
    """Assert that duplicate index entries can appear."""
    find(
        st_ak.indexed_option_array_contents(),
        lambda c: (
            len(c) >= 2
            and len(set(i for i in c.index.data if i >= 0))
            < sum(1 for i in c.index.data if i >= 0)
        ),
        settings=FIND,
    )


def test_draw_index_longer_than_content() -> None:
    """Assert that the index can be longer than the content."""
    find(
        st_ak.indexed_option_array_contents(),
        lambda c: len(c) > len(c.content),
        settings=FIND,
    )


def test_draw_from_contents() -> None:
    """Assert `contents()` can generate an `IndexedOptionArray` as outermost."""
    find(
        st_ak.contents(),
        lambda c: isinstance(c, IndexedOptionArray),
        settings=FIND,
    )
