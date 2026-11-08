import functools
from typing import Any, TypedDict, cast

import numpy as np
import pytest
from hypothesis import find, given
from hypothesis import strategies as st
from hypothesis.errors import Unsatisfiable

import awkward as ak
from awkward.contents import Content, IndexedArray, IndexedOptionArray, NumpyArray
from hypothesis_awkward import strategies as st_ak
from hypothesis_awkward._util import content_size, leaf_size
from hypothesis_awkward._util import safe_compare as sc
from hypothesis_awkward.strategies.constructors._types import StOption
from hypothesis_awkward.strategies.constructors.categorical import (
    categorical_from_contents,
)
from tests.find_settings import FIND, FIND_NO_SHRINK
from tests.funcs import (
    all_distinct,
    assert_kwargs_match_signature,
    awkward_can_check_unique,
)
from tests.record import RecordCallDraws
from tests.strategies.constructors.test_categorical import _has_duplicate_category


class CategoricalFromContentsKwargs(TypedDict, total=False):
    """Options for `categorical_from_contents()` strategy."""

    max_leaf_size: int | None
    min_length: int
    max_length: int | None
    allow_invalid: bool
    allow_indexed: bool
    allow_indexed_option: bool
    st_option: StOption | None


DEFAULTS = CategoricalFromContentsKwargs(
    max_leaf_size=None,
    min_length=0,
    max_length=None,
    allow_invalid=True,
    allow_indexed=True,
    allow_indexed_option=True,
    st_option=None,
)


class RelatedKwargs(TypedDict, total=False):
    """Options drawn together.

    `min_length <= max_length`, and at least one flavor must be allowed.
    """

    min_length: int
    max_length: int | None
    allow_indexed: bool
    allow_indexed_option: bool


def test_kwargs_match_signature() -> None:
    """Assert the option declarations agree with `categorical_from_contents()`."""
    assert_kwargs_match_signature(
        func=categorical_from_contents,
        exclude={'content', 'max_size'},
        kwargs_cls=CategoricalFromContentsKwargs,
        defaults=DEFAULTS,
        related_cls=RelatedKwargs,
    )


@st.composite
def categorical_from_contents_kwargs(draw: st.DrawFn) -> CategoricalFromContentsKwargs:
    """Strategy for options for `categorical_from_contents()` strategy."""

    @st.composite
    def _st_related_kwargs(draw: st.DrawFn) -> RelatedKwargs:
        """Strategy for the options that are drawn together."""
        min_length, max_length = draw(st_ak.misc.ranges(min_start=0, max_end=10))
        ret = RelatedKwargs()
        if min_length is not None:
            ret['min_length'] = min_length
        if max_length is not None:
            ret['max_length'] = max_length
        allowed = draw(
            st.sets(st.sampled_from(['indexed', 'indexed_option']), min_size=1)
        )
        ret['allow_indexed'] = 'indexed' in allowed
        ret['allow_indexed_option'] = 'indexed_option' in allowed
        return ret

    related = draw(_st_related_kwargs())

    optional_independent = cast(
        CategoricalFromContentsKwargs,
        draw(
            st.fixed_dictionaries(
                {},
                optional={
                    'max_leaf_size': st.one_of(
                        st.none(), st.integers(min_value=0, max_value=20)
                    ),
                    'allow_invalid': st.booleans(),
                    'st_option': st.none(),
                },
            )
        ),
    )

    return CategoricalFromContentsKwargs(**related, **optional_independent)


@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """Assert the results of `categorical_from_contents()`."""
    # Draw options
    content = RecordCallDraws(_leaves)
    max_size = data.draw(st.integers(min_value=0, max_value=50), label='max_size')
    kwargs = data.draw(categorical_from_contents_kwargs(), label='kwargs')

    # Call the test subject
    result = data.draw(
        categorical_from_contents(content, max_size=max_size, **kwargs), label='result'
    )

    assert isinstance(result, (IndexedArray, IndexedOptionArray))
    assert result.parameter('__array__') == 'categorical'

    # Assert the categories are what `content` drew
    categories = result.content
    assert any(categories is d for d in content.drawn)

    # Assert the options were effective
    assert content_size(result) <= max_size

    max_leaf_size = kwargs.get('max_leaf_size', DEFAULTS['max_leaf_size'])
    assert leaf_size(result) <= sc(max_leaf_size)

    min_length = kwargs.get('min_length', DEFAULTS['min_length'])
    max_length = kwargs.get('max_length', DEFAULTS['max_length'])
    assert min_length <= len(result) <= sc(max_length)

    allow_indexed = kwargs.get('allow_indexed', DEFAULTS['allow_indexed'])
    allow_indexed_option = kwargs.get(
        'allow_indexed_option', DEFAULTS['allow_indexed_option']
    )
    if not allow_indexed:
        assert not isinstance(result, IndexedArray)
    if not allow_indexed_option:
        assert not isinstance(result, IndexedOptionArray)

    allow_invalid = kwargs.get('allow_invalid', DEFAULTS['allow_invalid'])
    if not allow_invalid:
        if awkward_can_check_unique(categories):
            assert ak.validity_error(result) == ''
        else:
            assert isinstance(categories, NumpyArray)
            assert all_distinct(list(categories.data))


@pytest.mark.parametrize('type_', [IndexedArray, IndexedOptionArray])
def test_draw_flavor(type_: type[Content]) -> None:
    """Assert each flavor can be drawn."""
    find(_st_categorical(), lambda c: isinstance(c, type_), settings=FIND)


def test_draw_missing() -> None:
    """Assert a missing entry can be drawn."""
    find(_st_categorical(), _has_missing, settings=FIND)


def test_draw_missing_only() -> None:
    """Assert a non-empty index with no categories can be drawn."""
    find(_st_categorical(), lambda c: len(c) > 0 and len(c.content) == 0, settings=FIND)


def test_draw_max_size() -> None:
    """Assert content at exactly `max_size` can be drawn."""
    max_size = 20
    find(
        categorical_from_contents(_leaves, max_size=max_size),
        lambda c: content_size(c) == max_size,
        settings=FIND,
    )


def test_draw_max_leaf_size() -> None:
    """Assert content at exactly `max_leaf_size` can be drawn."""
    max_leaf_size = 5
    find(
        _st_categorical(max_leaf_size=max_leaf_size),
        lambda c: leaf_size(c) == max_leaf_size,
        settings=FIND,
    )


@pytest.mark.parametrize('min_length', [0, 1, 2, 5])
def test_draw_min_length(min_length: int) -> None:
    """Assert the length can reach `min_length`."""
    find(
        _st_categorical(min_length=min_length),
        lambda c: len(c) == min_length,
        settings=FIND,
    )


@pytest.mark.parametrize('max_length', [0, 1, 2, 5])
def test_draw_max_length(max_length: int) -> None:
    """Assert the length can reach `max_length`."""
    find(
        _st_categorical(max_length=max_length),
        lambda c: len(c) == max_length,
        settings=FIND,
    )


def test_draw_duplicate_categories() -> None:
    """Assert duplicate categories can be drawn by default."""
    find(
        _st_categorical(),
        lambda c: _has_duplicate_category(c.content),
        settings=FIND,
    )


def test_draw_distinct_categories_not_allow_invalid() -> None:
    """Assert two or more categories can be drawn if not `allow_invalid`."""
    find(
        _st_categorical(allow_invalid=False),
        lambda c: len(c.content) >= 2,
        settings=FIND,
    )


@pytest.mark.parametrize('max_size', [0, 30])
def test_raise_no_flavor(max_size: int) -> None:
    """Assert `ValueError` if no flavor is allowed, even if the lengths are infeasible.

    `min_length` exceeds `max_size` if `max_size` is 0.
    """
    with pytest.raises(ValueError):
        find(
            categorical_from_contents(
                _leaves,
                max_size=max_size,
                min_length=1,
                allow_indexed=False,
                allow_indexed_option=False,
            ),
            lambda _: True,
            settings=FIND_NO_SHRINK,
        )


def test_reject_invalid_kind() -> None:
    """Assert draws from `content` that are not valid as categories are rejected."""

    def content(**kwargs: Any) -> st.SearchStrategy[Content]:
        return st.just(NumpyArray(np.zeros((2, 2))))

    with pytest.raises(Unsatisfiable):
        find(
            categorical_from_contents(content, max_size=50),
            lambda _: True,
            settings=FIND_NO_SHRINK,
        )


def test_reject_non_unique_not_allow_invalid() -> None:
    """Assert non-unique draws from `content` are rejected if not `allow_invalid`."""

    def content(**kwargs: Any) -> st.SearchStrategy[Content]:
        return st.just(NumpyArray(np.array([1, 1])))

    with pytest.raises(Unsatisfiable):
        find(
            categorical_from_contents(content, max_size=50, allow_invalid=False),
            lambda _: True,
            settings=FIND_NO_SHRINK,
        )


def _has_missing(c: Content) -> bool:
    return isinstance(c, IndexedOptionArray) and bool((c.index.data < 0).any())


_leaves = functools.partial(
    st_ak.constructors.contents, max_depth=0, allow_record=False
)

_st_categorical = functools.partial(categorical_from_contents, _leaves, max_size=30)
