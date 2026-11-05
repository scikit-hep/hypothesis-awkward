from typing import Any, TypedDict, cast

import numpy as np
import pytest
from hypothesis import find, given
from hypothesis import strategies as st
from hypothesis.errors import Unsatisfiable

import awkward as ak
from awkward.contents import (
    Content,
    EmptyArray,
    IndexedArray,
    IndexedOptionArray,
    ListOffsetArray,
    NumpyArray,
    RecordArray,
)
from hypothesis_awkward import strategies as st_ak
from hypothesis_awkward.util import is_bytestring_leaf, is_string_leaf
from hypothesis_awkward.util import safe_compare as sc
from tests.find_settings import FIND, FIND_NO_SHRINK
from tests.funcs import KwargsDrawn, all_distinct, assert_kwargs_match_signature


class CategoricalContentsKwargs(TypedDict, total=False):
    """Options for `categorical_contents()` strategy."""

    content: st.SearchStrategy[Content] | Content | None
    allow_indexed: bool
    allow_indexed_option: bool
    min_size: int
    max_size: int | None


DEFAULTS = CategoricalContentsKwargs(
    content=None,
    allow_indexed=True,
    allow_indexed_option=True,
    min_size=0,
    max_size=None,
)

ALLOW_FLAGS = ('allow_indexed', 'allow_indexed_option')


class CategoricalContentsDrawn(TypedDict, total=False):
    """Values drawn in advance for the options of `categorical_contents()`."""

    content: Content


class RelatedKwargs(TypedDict, total=False):
    """Options drawn together.

    At least one flavor is allowed, and `min_size <= max_size`. They depend on the
    content drawn in advance: empty categories can be referenced only by `-1`
    entries, so `min_size` is 0 unless `IndexedOptionArray` is allowed.
    """

    allow_indexed: bool
    allow_indexed_option: bool
    min_size: int
    max_size: int | None


def test_kwargs_match_signature() -> None:
    """Assert the option declarations agree with `categorical_contents()`."""
    assert_kwargs_match_signature(
        func=st_ak.contents.categorical_contents,
        kwargs_cls=CategoricalContentsKwargs,
        defaults=DEFAULTS,
        related_cls=RelatedKwargs,
    )


@st.composite
def categorical_contents_kwargs(
    draw: st.DrawFn,
) -> KwargsDrawn[CategoricalContentsKwargs, CategoricalContentsDrawn]:
    """Strategy for options for `categorical_contents()` strategy."""

    @st.composite
    def _st_drawn(draw: st.DrawFn) -> CategoricalContentsDrawn:
        """Strategy for the values drawn in advance for the options."""
        ret = CategoricalContentsDrawn()
        if draw(st.booleans()):
            ret['content'] = draw(
                st_ak.contents.leaf_contents(unique=True, allow_zero_field_record=False)
            )
        return ret

    @st.composite
    def _st_related_kwargs(
        draw: st.DrawFn, drawn: CategoricalContentsDrawn
    ) -> RelatedKwargs:
        """Strategy for the options that are drawn together."""
        ret = dict[str, Any]()
        allowed = draw(st.sets(st.sampled_from(ALLOW_FLAGS), min_size=1))
        for flag in ALLOW_FLAGS:
            # A disallowed flag must be passed; an allowed one may be omitted.
            if flag not in allowed or draw(st.booleans()):
                ret[flag] = flag in allowed
        max_start = 50
        if (
            'content' in drawn
            and len(drawn['content']) == 0
            and 'allow_indexed_option' not in allowed
        ):
            max_start = 0  # an empty index is the only one possible
        min_size, max_size = draw(
            st_ak.ranges(min_start=0, max_start=max_start, max_end=50)
        )
        if min_size is not None:
            ret['min_size'] = min_size
        if max_size is not None:
            ret['max_size'] = max_size
        return cast(RelatedKwargs, ret)

    drawn = draw(_st_drawn())
    related = draw(_st_related_kwargs(drawn))

    optional_independent = CategoricalContentsKwargs()
    if 'content' in drawn:
        content = drawn['content']
        optional_independent['content'] = draw(
            st.sampled_from([content, st.just(content)])
        )
    elif draw(st.booleans()):
        optional_independent['content'] = None

    kwargs = CategoricalContentsKwargs(**related, **optional_independent)
    return KwargsDrawn(kwargs, drawn)


@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """Assert the results of `categorical_contents()`."""
    # Draw options
    kwargs, drawn = data.draw(categorical_contents_kwargs(), label='opts')

    # Call the test subject
    result = data.draw(st_ak.contents.categorical_contents(**kwargs), label='result')

    assert isinstance(result, (IndexedArray, IndexedOptionArray))
    assert result.parameter('__array__') == 'categorical'

    # Assert the layout is valid; Awkward checks that the categories are unique
    assert ak.validity_error(result) == ''

    # Assert the categories are a leaf with no two elements equal
    categories = result.content
    match categories:
        case NumpyArray():
            assert categories.data.ndim == 1
            assert all_distinct(list(categories.data))
        case EmptyArray():
            pass
        case _:
            assert is_string_leaf(categories) or is_bytestring_leaf(categories)
            assert all_distinct(ak.Array(categories).to_list())

    # Assert all index entries refer to a category or are missing (-1)
    index = result.index.data
    lowest = -1 if isinstance(result, IndexedOptionArray) else 0
    assert all(lowest <= i < len(categories) for i in index.tolist())

    # Assert the options were effective
    allow_indexed = kwargs.get('allow_indexed', DEFAULTS['allow_indexed'])
    allow_indexed_option = kwargs.get(
        'allow_indexed_option', DEFAULTS['allow_indexed_option']
    )
    if not allow_indexed:
        assert not isinstance(result, IndexedArray)
    if not allow_indexed_option:
        assert not isinstance(result, IndexedOptionArray)

    min_size = kwargs.get('min_size', DEFAULTS['min_size'])
    max_size = kwargs.get('max_size', DEFAULTS['max_size'])
    assert min_size <= len(result) <= sc(max_size)

    if 'content' in drawn:
        assert result.content is drawn['content']


@pytest.mark.parametrize('type_', [IndexedArray, IndexedOptionArray])
def test_draw_flavor(type_: type[Content]) -> None:
    """Assert each flavor can be drawn by default."""
    find(
        st_ak.contents.categorical_contents(),
        lambda c: isinstance(c, type_),
        settings=FIND,
    )


def test_draw_missing() -> None:
    """Assert a missing entry (-1) can be drawn."""
    find(
        st_ak.contents.categorical_contents(),
        lambda c: isinstance(c, IndexedOptionArray) and -1 in c.index.data.tolist(),
        settings=FIND,
    )


@pytest.mark.parametrize('min_size', [1, 2, 10])
def test_draw_min_size(min_size: int) -> None:
    """Assert the index length can reach `min_size`."""
    find(
        st_ak.contents.categorical_contents(min_size=min_size),
        lambda c: len(c) == min_size,
        settings=FIND,
    )


@pytest.mark.parametrize(
    'is_kind',
    [
        pytest.param(lambda c: isinstance(c, NumpyArray), id='numpy'),
        pytest.param(lambda c: isinstance(c, EmptyArray), id='empty'),
        pytest.param(is_string_leaf, id='string'),
        pytest.param(is_bytestring_leaf, id='bytestring'),
    ],
)
def test_draw_categories(is_kind: Any) -> None:
    """Assert each kind of categories can be drawn by default."""
    find(
        st_ak.contents.categorical_contents(),
        lambda c: is_kind(c.content),
        settings=FIND,
    )


def test_draw_repeated_category() -> None:
    """Assert a category referred to by more than one entry can be drawn."""
    find(
        st_ak.contents.categorical_contents(),
        lambda c: _has_repeated_category(c),
        settings=FIND,
    )


def test_draw_unused_category() -> None:
    """Assert a category referred to by no entry can be drawn.

    Slicing categorical data keeps all categories, so unused ones are common.
    """
    find(
        st_ak.contents.categorical_contents(),
        lambda c: _has_unused_category(c),
        settings=FIND,
    )


def test_raise_no_flavor() -> None:
    """Assert `ValueError` if both flavors are disallowed."""
    with pytest.raises(ValueError):
        find(
            st_ak.contents.categorical_contents(
                allow_indexed=False, allow_indexed_option=False
            ),
            lambda _: True,
            settings=FIND_NO_SHRINK,
        )


@pytest.mark.parametrize(
    'content',
    [
        pytest.param(NumpyArray(np.array([1, 2, 1])), id='non-unique'),
        pytest.param(NumpyArray(np.array([[1, 2], [3, 4]])), id='2d'),
        pytest.param(
            ListOffsetArray(
                ak.index.Index64(np.array([0, 1, 3])), NumpyArray(np.array([1, 2, 3]))
            ),
            id='list',
        ),
        pytest.param(RecordArray([], fields=[], length=1), id='zero-field-record'),
    ],
)
def test_raise_invalid_content(content: Content) -> None:
    """Assert `ValueError` if a concrete `content` is not valid categories."""
    with pytest.raises(ValueError):
        find(
            st_ak.contents.categorical_contents(content),
            lambda _: True,
            settings=FIND_NO_SHRINK,
        )


def test_reject_invalid_content_from_strategy() -> None:
    """Assert draws of invalid categories from a `content` strategy are rejected."""
    content = NumpyArray(np.array([1, 2, 1]))
    with pytest.raises(Unsatisfiable):
        find(
            st_ak.contents.categorical_contents(st.just(content)),
            lambda _: True,
            settings=FIND_NO_SHRINK,
        )


def _has_repeated_category(c: Content) -> bool:
    entries = [i for i in cast(Any, c).index.data.tolist() if i >= 0]
    return len(set(entries)) < len(entries)


def _has_unused_category(c: Content) -> bool:
    entries = {i for i in cast(Any, c).index.data.tolist() if i >= 0}
    return len(entries) < len(cast(Any, c).content)
