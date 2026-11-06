import itertools
from typing import TypedDict, cast

import numpy as np
import pytest
from hypothesis import find, given
from hypothesis import strategies as st
from hypothesis.errors import Unsatisfiable

import awkward as ak
from awkward._do import mergeable
from awkward.contents import (
    Content,
    EmptyArray,
    IndexedOptionArray,
    NumpyArray,
    UnionArray,
)
from hypothesis_awkward import strategies as st_ak
from hypothesis_awkward.util import iter_contents
from hypothesis_awkward.util import safe_compare as sc
from tests.find_settings import FIND, FIND_NO_SHRINK, FIND_RARE_NO_SHRINK
from tests.funcs import KwargsDrawn, assert_kwargs_match_signature


class UnionArrayContentsKwargs(TypedDict, total=False):
    """Options for `union_array_contents()` strategy."""

    contents: list[Content] | st.SearchStrategy[list[Content]] | None
    max_contents: int
    min_length: int
    max_length: int | None
    allow_invalid: bool


DEFAULTS = UnionArrayContentsKwargs(
    contents=None,
    max_contents=4,
    min_length=0,
    max_length=None,
    allow_invalid=True,
)


class UnionArrayContentsDrawn(TypedDict, total=False):
    """Values drawn in advance for the options of `union_array_contents()`."""

    allow_invalid: bool
    contents: list[Content]


class RelatedKwargs(TypedDict, total=False):
    """Options drawn together as an ordered pair (`min_length <= max_length`)."""

    min_length: int
    max_length: int | None


def test_kwargs_match_signature() -> None:
    """Assert the option declarations agree with `union_array_contents()`."""
    assert_kwargs_match_signature(
        func=st_ak.contents.union_array_contents,
        kwargs_cls=UnionArrayContentsKwargs,
        defaults=DEFAULTS,
        related_cls=RelatedKwargs,
    )


@st.composite
def _contents_list(draw: st.DrawFn, *, allow_invalid: bool) -> list[Content]:
    """Draw a list of Content objects for testing.

    No two of them are mergeable if `allow_invalid` is `False`.
    """
    n = draw(st.integers(min_value=0, max_value=5))
    ret = list[Content]()
    for _ in range(n):
        c = draw(
            st_ak.contents.contents(
                max_leaf_size=5,
                max_depth=2,
                allow_union=False,
                allow_option_root=False,
                allow_indexed_root=False,
            )
        )
        if allow_invalid or not any(_is_mergeable(r, c) for r in ret):
            ret.append(c)
    return ret


@st.composite
def union_array_contents_kwargs(
    draw: st.DrawFn,
) -> KwargsDrawn[UnionArrayContentsKwargs, UnionArrayContentsDrawn]:
    """Strategy for options for `union_array_contents()` strategy."""

    @st.composite
    def _st_related_kwargs(draw: st.DrawFn) -> RelatedKwargs:
        """Strategy for the options that are drawn together."""
        min_length, max_length = draw(st_ak.ranges(min_start=0, max_end=10))
        ret = RelatedKwargs()
        if min_length is not None:
            ret['min_length'] = min_length
        if max_length is not None:
            ret['max_length'] = max_length
        return ret

    @st.composite
    def _st_drawn(draw: st.DrawFn) -> UnionArrayContentsDrawn:
        """Strategy for the values drawn in advance for the options.

        The content list depends on `allow_invalid`: if `False`, it has no two
        mergeable contents.
        """
        ret = UnionArrayContentsDrawn()
        if draw(st.booleans()):
            ret['allow_invalid'] = draw(st.booleans())
        allow_invalid = ret.get('allow_invalid', DEFAULTS['allow_invalid'])
        if draw(st.booleans()):
            ret['contents'] = draw(_contents_list(allow_invalid=allow_invalid))
        return ret

    @st.composite
    def _st_optional_independent(
        draw: st.DrawFn, drawn: UnionArrayContentsDrawn
    ) -> UnionArrayContentsKwargs:
        """Strategy for the options that are drawn independently, if at all."""
        ret = cast(
            UnionArrayContentsKwargs,
            draw(
                st.fixed_dictionaries(
                    {},
                    optional={'max_contents': st.integers(min_value=0, max_value=10)},
                )
            ),
        )
        if 'contents' in drawn:
            contents = drawn['contents']
            choices = list[list[Content] | st.SearchStrategy[list[Content]]](
                [contents, st.just(contents)]
            )
            ret['contents'] = draw(st.sampled_from(choices))
        if 'allow_invalid' in drawn:
            ret['allow_invalid'] = drawn['allow_invalid']
        return ret

    related = draw(_st_related_kwargs())
    drawn = draw(_st_drawn())
    optional_independent = draw(_st_optional_independent(drawn))

    kwargs = UnionArrayContentsKwargs(**related, **optional_independent)
    return KwargsDrawn(kwargs, drawn)


@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """Assert the results of `union_array_contents()`."""
    # Draw options
    kwargs, drawn = data.draw(union_array_contents_kwargs(), label='opts')

    max_contents = kwargs.get('max_contents', DEFAULTS['max_contents'])

    # `UnionArray` requires at least two contents.
    if 'contents' in kwargs:
        expect_raised = len(drawn['contents']) < 2
    else:
        expect_raised = max_contents < 2

    # Call the test subject
    try:
        result = data.draw(
            st_ak.contents.union_array_contents(**kwargs), label='result'
        )
    except TypeError as e:
        assert expect_raised
        assert 'at least 2' in str(e)
        return

    assert not expect_raised

    assert isinstance(result, UnionArray)

    # Assert the options were effective
    if 'contents' in kwargs:
        # Given contents: the contents are the same objects
        assert len(result.contents) == len(drawn['contents'])
        for r_content, drawn_content in zip(result.contents, drawn['contents']):
            assert r_content is drawn_content
    else:
        assert len(result.contents) <= max_contents

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
    max_length = kwargs.get('max_length', DEFAULTS['max_length'])
    if max_length is None:
        for tag_val in range(len(result.contents)):
            indices_for_tag = index[tags == tag_val]
            content_len = len(result.contents[tag_val])
            assert len(indices_for_tag) == content_len
            assert set(indices_for_tag.tolist()) == set(range(content_len))

    # Assert length is within bounds
    min_length = kwargs.get('min_length', DEFAULTS['min_length'])
    assert min_length <= len(result) <= sc(max_length)

    allow_invalid = kwargs.get('allow_invalid', DEFAULTS['allow_invalid'])
    if not allow_invalid:
        assert ak.validity_error(result) == ''


@given(u=st_ak.contents.union_array_contents(allow_invalid=False))
def test_valid_not_allow_invalid(u: UnionArray) -> None:
    """Assert the union is valid if not `allow_invalid`, with no other options."""
    assert ak.validity_error(u) == ''


@pytest.mark.parametrize('max_contents', [2, 3, 4])
def test_draw_max_contents(max_contents: int) -> None:
    """Assert the content count can reach `max_contents`."""
    find(
        st_ak.contents.union_array_contents(max_contents=max_contents),
        lambda u: len(u.contents) == max_contents,
        settings=FIND,
    )


def test_draw_different_content_lengths() -> None:
    """Assert the contents can have different lengths."""
    find(
        st_ak.contents.union_array_contents(),
        lambda u: len({len(c) for c in u.contents}) > 1,
        settings=FIND,
    )


@pytest.mark.parametrize('min_length', [1, 2, 10])
def test_draw_min_length(min_length: int) -> None:
    """Assert the length can reach `min_length`."""
    find(
        st_ak.contents.union_array_contents(min_length=min_length),
        lambda u: len(u) == min_length,
        settings=FIND,
    )


@pytest.mark.parametrize('max_length', [1, 2, 10])
def test_draw_max_length(max_length: int) -> None:
    """Assert the length can reach `max_length`."""
    find(
        st_ak.contents.union_array_contents(max_length=max_length),
        lambda u: len(u) == max_length,
        settings=FIND,
    )


def test_draw_mergeable_contents() -> None:
    """Assert mergeable child contents can be drawn by default."""
    find(
        st_ak.contents.union_array_contents(),
        lambda u: any(
            _is_mergeable(one, two)
            for one, two in itertools.combinations(u.contents, 2)
        ),
        settings=FIND,
    )


@pytest.mark.parametrize('max_contents', [2, 3])
def test_draw_max_contents_not_allow_invalid(max_contents: int) -> None:
    """Assert the content count can reach `max_contents` if not `allow_invalid`."""
    find(
        st_ak.contents.union_array_contents(
            max_contents=max_contents, allow_invalid=False
        ),
        lambda u: len(u.contents) == max_contents,
        settings=FIND,
    )


def test_draw_all_option_not_allow_invalid() -> None:
    """Assert all-option children can be drawn if not `allow_invalid`."""
    find(
        st_ak.contents.union_array_contents(allow_invalid=False),
        lambda u: all(child.is_option for child in u.contents),
        settings=FIND_RARE_NO_SHRINK,
    )


@pytest.mark.parametrize(
    'contents',
    [
        pytest.param([EmptyArray(), NumpyArray(np.array([1]))], id='empty-numpy'),
        pytest.param(
            [NumpyArray(np.array([1])), NumpyArray(np.array([1.0]))], id='int-float'
        ),
        pytest.param(
            [
                NumpyArray(np.array([True])),
                NumpyArray(np.array([1])),
                NumpyArray(np.array([1.0])),
            ],
            id='last-two',
        ),
        pytest.param(
            [
                IndexedOptionArray(
                    ak.index.Index64(np.array([0])), NumpyArray(np.array([1]))
                ),
                IndexedOptionArray(
                    ak.index.Index64(np.array([0])), NumpyArray(np.array([1.0]))
                ),
            ],
            id='option',
        ),
    ],
)
def test_raise_mergeable_contents(contents: list[Content]) -> None:
    """Assert `ValueError` if a concrete `contents` has mergeable child contents."""
    with pytest.raises(ValueError):
        find(
            st_ak.contents.union_array_contents(contents, allow_invalid=False),
            lambda _: True,
            settings=FIND_NO_SHRINK,
        )


def test_reject_mergeable_contents_from_strategy() -> None:
    """Assert draws of mergeable child contents from a strategy are rejected."""
    contents = list[Content]([EmptyArray(), NumpyArray(np.array([1]))])
    with pytest.raises(Unsatisfiable):
        find(
            st_ak.contents.union_array_contents(st.just(contents), allow_invalid=False),
            lambda _: True,
            settings=FIND_NO_SHRINK,
        )


def test_draw_from_contents() -> None:
    """Assert `contents()` can generate a `UnionArray` as outermost."""
    find(st_ak.contents.contents(), lambda c: isinstance(c, UnionArray), settings=FIND)


def test_draw_nested_union() -> None:
    """Assert that a UnionArray with a descendant UnionArray can be drawn.

    Direct children of a union must not be unions (no `Union[Union[...]]`),
    but deeper descendants may be, e.g. `Union[Record[Union[...], ...], ...]`.
    """
    find(
        st_ak.contents.union_array_contents(),
        _has_nested_union,
        settings=FIND_RARE_NO_SHRINK,
    )


def test_draw_from_contents_nested_union() -> None:
    """Assert that a UnionArray with a descendant UnionArray can be drawn.

    Direct children of a union must not be unions (no `Union[Union[...]]`),
    but deeper descendants may be, e.g. `Union[Record[Union[...], ...], ...]`.
    """
    find(
        st_ak.contents.contents(max_leaf_size=20, max_depth=5),
        _has_nested_union,
        settings=FIND_RARE_NO_SHRINK,
    )


@pytest.mark.parametrize('nested', [True, False])
def test_draw_from_contents_not_allow_invalid(nested: bool) -> None:
    """Assert a `UnionArray` at each position can be drawn if not `allow_invalid`."""
    find(
        st_ak.contents.contents(allow_invalid=False),
        lambda c: any(
            isinstance(n, UnionArray) and (n is not c) == nested
            for n in iter_contents(c)
        ),
        settings=FIND,
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
    descendants are, e.g., `Union[ListOffset[ByteMasked[...]], ...]`.
    """
    find(
        st_ak.contents.contents(max_leaf_size=20, max_depth=5),
        _has_option_deep_inside_union,
        settings=FIND_RARE_NO_SHRINK,
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
        settings=FIND_RARE_NO_SHRINK,
    )


def test_draw_all_option_union() -> None:
    """Assert that standalone union_array_contents() can produce all-option children."""
    find(
        st_ak.contents.union_array_contents(),
        lambda c: all(child.is_option for child in c.contents),
        settings=FIND_RARE_NO_SHRINK,
    )


def _has_all_option_union(c: Content) -> bool:
    for node in iter_contents(c):
        if not isinstance(node, UnionArray):
            continue
        if all(child.is_option for child in node.contents):
            return True
    return False


def _is_mergeable(one: Content, two: Content) -> bool:
    return bool(mergeable(one, two, mergebool=False))
