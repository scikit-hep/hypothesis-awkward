from typing import Any, TypedDict, cast

import numpy as np
import pytest
from hypothesis import Phase, find, given, settings
from hypothesis import strategies as st

from awkward.contents import Content, UnionArray
from hypothesis_awkward import strategies as st_ak
from hypothesis_awkward.util import iter_contents
from hypothesis_awkward.util import safe_compare as sc

DEFAULT_MAX_CONTENTS = 4


class UnionArrayContentsKwargs(TypedDict, total=False):
    """Options for `union_array_contents()` strategy."""

    contents: list[Content] | st.SearchStrategy[list[Content]]
    max_contents: int
    max_length: int


@st.composite
def _contents_list(
    draw: st.DrawFn,
) -> list[Content]:
    """Draw a list of 2..5 Content objects for testing."""
    n = draw(st.integers(min_value=2, max_value=5))
    return [
        draw(
            st_ak.contents.contents(
                max_leaf_size=5,
                max_depth=2,
                allow_union=False,
                allow_option_root=False,
            )
        )
        for _ in range(n)
    ]


@st.composite
def union_array_contents_kwargs(
    draw: st.DrawFn,
    chain: st_ak.OptsChain[Any] | None = None,
) -> st_ak.OptsChain[UnionArrayContentsKwargs]:
    """Strategy for options for `union_array_contents()` strategy."""
    if chain is None:
        chain = st_ak.OptsChain({})
    st_contents = chain.register(_contents_list())

    kwargs = draw(
        st.fixed_dictionaries(
            {},
            optional={
                'contents': st.one_of(
                    _contents_list(),
                    st.just(st_contents),
                ),
                'max_contents': st.integers(min_value=2, max_value=10),
                'max_length': st.integers(min_value=0, max_value=50),
            },
        )
    )

    return chain.extend(cast(UnionArrayContentsKwargs, kwargs))


@settings(max_examples=200)
@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """Assert the results of `union_array_contents()`."""
    # Draw options
    opts = data.draw(union_array_contents_kwargs(), label='opts')
    opts.reset()

    # Call the test subject
    result = data.draw(
        st_ak.contents.union_array_contents(**opts.kwargs), label='result'
    )

    assert isinstance(result, UnionArray)

    # Assert the options were effective
    max_contents = opts.kwargs.get('max_contents', DEFAULT_MAX_CONTENTS)
    contents = opts.kwargs.get('contents', None)

    # Contents dispatch
    match contents:
        case None:
            assert len(result.contents) <= max_contents
        case list():
            # Concrete list: contents should be the same objects
            assert len(result.contents) == len(contents)
            for r_content, given_content in zip(result.contents, contents):
                assert r_content is given_content
        case st_ak.RecordDraws():
            assert len(contents.drawn) == 1
            drawn_list = contents.drawn[0]
            assert len(result.contents) == len(drawn_list)
            for r_content, drawn_content in zip(result.contents, drawn_list):
                assert r_content is drawn_content

    # Tags validity: dtype is int8, values in [0, len(contents))
    tags = result.tags.data
    assert tags.dtype == np.int8
    assert np.all(tags >= 0)
    assert np.all(tags < len(result.contents))

    # Index validity: dtype in {int32, uint32, int64}, values in valid range
    index = result.index.data
    assert index.dtype in (np.int32, np.uint32, np.int64)
    for tag_val in range(len(result.contents)):
        content_len = len(result.contents[tag_val])
        indices_for_tag = index[tags == tag_val]
        if len(indices_for_tag) > 0:
            assert np.all(indices_for_tag >= 0)
            assert np.all(indices_for_tag < content_len)

    # Length: tags and index have the same length
    assert len(tags) == len(index)
    assert len(result) == len(tags)

    # Compact indexing holds unless max_length is given
    max_length = opts.kwargs.get('max_length')
    if max_length is None:
        for tag_val in range(len(result.contents)):
            indices_for_tag = index[tags == tag_val]
            content_len = len(result.contents[tag_val])
            assert len(indices_for_tag) == content_len
            assert set(indices_for_tag.tolist()) == set(range(content_len))

    # Assert max_length constraint
    assert len(result) <= sc(max_length)


@pytest.mark.parametrize('max_contents', [2, 3, 4])
def test_draw_max_contents(max_contents: int) -> None:
    """Assert the content count can reach `max_contents`."""
    find(
        st_ak.contents.union_array_contents(max_contents=max_contents),
        lambda u: len(u.contents) == max_contents,
    )


def test_draw_different_content_lengths() -> None:
    """Assert the contents can have different lengths."""
    find(
        st_ak.contents.union_array_contents(),
        lambda u: len({len(c) for c in u.contents}) > 1,
    )


@pytest.mark.parametrize('max_length', [1, 2, 10])
def test_draw_max_length(max_length: int) -> None:
    """Assert the length can reach `max_length`."""
    find(
        st_ak.contents.union_array_contents(max_length=max_length),
        lambda u: len(u) == max_length,
    )


def test_draw_from_contents() -> None:
    """Assert `contents()` can generate a `UnionArray` as outermost."""
    find(
        st_ak.contents.contents(),
        lambda c: isinstance(c, UnionArray),
        settings=settings(max_examples=2000),
    )


def test_draw_nested_union() -> None:
    """Assert that a UnionArray with a descendant UnionArray can be drawn.

    Direct children of a union must not be unions (no ``Union[Union[...]]``),
    but deeper descendants may be, e.g. ``Union[Record[Union[...], ...], ...]``.
    """
    find(
        st_ak.contents.union_array_contents(),
        _has_nested_union,
        settings=settings(phases=[Phase.generate], max_examples=5000),
    )


def test_draw_from_contents_nested_union() -> None:
    """Assert that a UnionArray with a descendant UnionArray can be drawn.

    Direct children of a union must not be unions (no ``Union[Union[...]]``),
    but deeper descendants may be, e.g. ``Union[Record[Union[...], ...], ...]``.
    """
    find(
        st_ak.contents.contents(max_leaf_size=20, max_depth=5),
        _has_nested_union,
        settings=settings(phases=[Phase.generate], max_examples=5000),
    )


def _has_nested_union(c: Content) -> bool:
    for node in iter_contents(c):
        if not isinstance(node, UnionArray):
            continue
        # Check if any descendant is a UnionArray
        for child in node.contents:
            for descendant in iter_contents(child):
                if isinstance(descendant, UnionArray):
                    return True
    return False


def test_draw_from_contents_option_deep_inside_union() -> None:
    """Assert that option types can appear deep inside non-option union branches.

    The direct children of the union are not option types, but deeper
    descendants are, e.g., ``Union[ListOffset[ByteMasked[...]], ...]``.
    """
    find(
        st_ak.contents.contents(max_leaf_size=20, max_depth=5),
        _has_option_deep_inside_union,
        settings=settings(phases=[Phase.generate], max_examples=5000),
    )


def _has_option_deep_inside_union(c: Content) -> bool:
    for node in iter_contents(c):
        if not isinstance(node, UnionArray):
            continue
        for child in node.contents:
            if child.is_option:
                continue
            if any(d.is_option for d in iter_contents(child)):
                return True
    return False


def test_draw_from_contents_all_option_union() -> None:
    """Assert that a UnionArray with all-option children can be drawn.

    All direct children of the union are option types, satisfying the "all or none"
    rule.
    """
    find(
        st_ak.contents.contents(max_leaf_size=20, max_depth=5),
        _has_all_option_union,
        settings=settings(phases=[Phase.generate], max_examples=5000),
    )


def test_draw_all_option_union() -> None:
    """Assert that standalone union_array_contents() can produce all-option children."""
    find(
        st_ak.contents.union_array_contents(),
        lambda c: all(child.is_option for child in c.contents),
        settings=settings(phases=[Phase.generate], max_examples=5000),
    )


def _has_all_option_union(c: Content) -> bool:
    for node in iter_contents(c):
        if not isinstance(node, UnionArray):
            continue
        if all(child.is_option for child in node.contents):
            return True
    return False
