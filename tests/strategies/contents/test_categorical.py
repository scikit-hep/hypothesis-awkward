import functools
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
from hypothesis_awkward.util import is_bytestring_leaf, is_string_leaf, iter_contents
from hypothesis_awkward.util import safe_compare as sc
from tests.find_settings import FIND, FIND_NO_SHRINK, FIND_RARE
from tests.funcs import (
    KwargsDrawn,
    all_distinct,
    assert_kwargs_match_signature,
    awkward_can_check_unique,
)


class CategoricalContentsKwargs(TypedDict, total=False):
    """Options for `categorical_contents()` strategy."""

    content: st.SearchStrategy[Content] | Content | None
    min_size: int
    max_size: int | None
    allow_invalid: bool
    allow_indexed: bool
    allow_indexed_option: bool


DEFAULTS = CategoricalContentsKwargs(
    content=None,
    min_size=0,
    max_size=None,
    allow_invalid=True,
    allow_indexed=True,
    allow_indexed_option=True,
)

ALLOW_FLAGS = ('allow_indexed', 'allow_indexed_option')


class CategoricalContentsDrawn(TypedDict, total=False):
    """Values drawn in advance for the options of `categorical_contents()`."""

    allow_invalid: bool
    content: Content


class RelatedKwargs(TypedDict, total=False):
    """Options drawn together.

    At least one flavor is allowed, and `min_size <= max_size`. They depend on the
    content drawn in advance: empty categories can be referenced only by `-1`
    entries, so `min_size` is 0 unless `IndexedOptionArray` is allowed.
    """

    min_size: int
    max_size: int | None
    allow_indexed: bool
    allow_indexed_option: bool


_st_contents_categorical = functools.partial(
    st_ak.contents.contents,
    allow_regular=False,
    allow_list_offset=False,
    allow_list=False,
    allow_record=False,
    allow_union=False,
    allow_byte_masked=False,
    allow_bit_masked=False,
    allow_unmasked=False,
)


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
        """Strategy for the values drawn in advance for the options.

        The content depends on `allow_invalid`: if `False`, it has no two categories
        equal as Awkward Array defines it, which allows at most one `NaT`.
        """
        ret = CategoricalContentsDrawn()
        if draw(st.booleans()):
            ret['allow_invalid'] = draw(st.booleans())
        allow_invalid = ret.get('allow_invalid', DEFAULTS['allow_invalid'])
        if draw(st.booleans()):
            leaf = st_ak.contents.leaf_contents(
                unique=not allow_invalid, allow_zero_field_record=False
            )
            if not allow_invalid:
                leaf = leaf.filter(lambda c: _n_nat(c) <= 1)
            ret['content'] = draw(leaf)
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
    if 'allow_invalid' in drawn:
        optional_independent['allow_invalid'] = drawn['allow_invalid']
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

    # Assert the categories are a 1-D NumPy array, empty, strings, or bytestrings
    categories = result.content
    match categories:
        case NumpyArray():
            assert categories.data.ndim == 1
        case EmptyArray():
            pass
        case _:
            assert is_string_leaf(categories) or is_bytestring_leaf(categories)

    # Assert all index entries refer to a category or are missing (-1)
    index = result.index.data
    lowest = -1 if isinstance(result, IndexedOptionArray) else 0
    assert all(lowest <= i < len(categories) for i in index.tolist())

    # Assert the options were effective
    allow_invalid = kwargs.get('allow_invalid', DEFAULTS['allow_invalid'])
    if not allow_invalid:
        if awkward_can_check_unique(categories):
            # Awkward checks that the categories are unique
            assert ak.validity_error(result) == ''
        else:
            # Without `NaT`, `==` agrees with Awkward's definition of unique.
            assert isinstance(categories, NumpyArray)
            assert all_distinct(list(categories.data))

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


@pytest.mark.parametrize('allow_invalid', [False, True])
@pytest.mark.parametrize(
    'is_kind',
    [
        pytest.param(lambda c: isinstance(c, NumpyArray), id='numpy'),
        pytest.param(lambda c: isinstance(c, EmptyArray), id='empty'),
        pytest.param(is_string_leaf, id='string'),
        pytest.param(is_bytestring_leaf, id='bytestring'),
    ],
)
def test_draw_categories(is_kind: Any, allow_invalid: bool) -> None:
    """Assert each kind of categories can be drawn."""
    find(
        st_ak.contents.categorical_contents(allow_invalid=allow_invalid),
        lambda c: is_kind(c.content),
        settings=FIND,
    )


def test_draw_duplicate_categories() -> None:
    """Assert categories with two equal elements can be drawn by default.

    Awkward Array builds and loads such categorical data without an error, although
    `ak.is_valid()` returns `False` for it.
    """
    find(
        st_ak.contents.categorical_contents(),
        lambda c: _has_duplicate_category(c.content),
        settings=FIND,
    )


def test_draw_nat() -> None:
    """Assert a `NaT` category can be drawn under `allow_invalid=False`."""
    find(
        st_ak.contents.categorical_contents(allow_invalid=False),
        lambda c: _n_nat(c.content) == 1,
        settings=FIND,
    )


def test_draw_multiple_nan() -> None:
    """Assert more than one `NaN` category can be drawn under `allow_invalid=False`.

    Awkward Array counts `NaN`s as distinct, unlike `NaT`s.
    """
    find(
        st_ak.contents.categorical_contents(allow_invalid=False),
        lambda c: _n_nan(c.content) >= 2,
        settings=FIND_RARE,
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


def test_draw_from_contents() -> None:
    """Assert `contents()` can generate categorical data as outermost."""
    find(st_ak.contents.contents(), _is_categorical, settings=FIND)


@pytest.mark.parametrize(
    'type_, flag',
    [
        pytest.param(IndexedArray, 'allow_indexed_option', id='indexed'),
        pytest.param(IndexedOptionArray, 'allow_indexed', id='indexed-option'),
    ],
)
def test_draw_from_contents_one_flavor(type_: type[Content], flag: str) -> None:
    """Assert `contents()` can generate a flavor if the other one is not allowed."""
    kwargs = cast(dict[str, Any], {flag: False})
    find(
        st_ak.contents.contents(**kwargs),
        lambda c: isinstance(c, type_) and _is_categorical(c),
        settings=FIND,
    )


def test_draw_from_contents_categorical_root() -> None:
    """Assert `allow_categorical_root` overrides `allow_indexed_root`."""
    find(
        _st_contents_categorical(
            allow_indexed_option=False,
            allow_indexed_root=False,
            allow_categorical_root=True,
        ),
        _is_categorical,
        settings=FIND,
    )


def test_draw_from_contents_nested() -> None:
    """Assert `contents()` can generate categorical data below the outermost node."""
    find(
        st_ak.contents.contents(),
        lambda c: any(n is not c and _is_categorical(n) for n in iter_contents(c)),
        settings=FIND,
    )


def test_draw_from_contents_duplicate_categories() -> None:
    """Assert `contents()` can generate duplicate categories by default.

    Categories with `NaT` are excluded: `NaT`s can repeat even in unique draws.
    """
    find(
        st_ak.contents.contents(),
        lambda c: (
            _is_categorical(c)
            and _has_duplicate_category(c.content)
            and not _has_nat(c.content)
        ),
        settings=FIND,
    )


def test_draw_from_contents_not_allow_invalid() -> None:
    """Assert `contents()` can generate categories if not `allow_invalid`."""
    find(
        st_ak.contents.contents(allow_invalid=False),
        lambda c: _is_categorical(c) and len(c.content) >= 2,
        settings=FIND,
    )


@given(c=_st_contents_categorical(allow_invalid=False, allow_indexed=False))
def test_from_contents_distinct_not_allow_invalid(c: Content) -> None:
    """Assert `contents()` passes `allow_invalid=False` on to categorical data."""
    assert not any(
        _has_duplicate_category(n.content)
        for n in iter_contents(c)
        if _is_categorical(n)
    )


@given(c=_st_contents_categorical(allow_indexed=False))
def test_from_contents_not_allow_indexed(c: Content) -> None:
    """Assert `contents()` generates no `IndexedArray` if not `allow_indexed`."""
    assert not any(isinstance(n, IndexedArray) for n in iter_contents(c))


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


@pytest.mark.parametrize('allow_invalid', [False, True])
@pytest.mark.parametrize(
    'content',
    [
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
def test_raise_invalid_kind(content: Content, allow_invalid: bool) -> None:
    """Assert `ValueError` if a concrete `content` is not a kind of categories."""
    with pytest.raises(ValueError):
        find(
            st_ak.contents.categorical_contents(content, allow_invalid=allow_invalid),
            lambda _: True,
            settings=FIND_NO_SHRINK,
        )


@pytest.mark.parametrize(
    'content',
    [
        pytest.param(NumpyArray(np.array([1, 2, 1])), id='duplicate'),
        pytest.param(
            NumpyArray(np.array(['NaT', 'NaT'], dtype='datetime64[s]')), id='two-nat'
        ),
    ],
)
def test_raise_non_unique_content(content: Content) -> None:
    """Assert `ValueError` if a concrete `content` is not unique."""
    with pytest.raises(ValueError):
        find(
            st_ak.contents.categorical_contents(content, allow_invalid=False),
            lambda _: True,
            settings=FIND_NO_SHRINK,
        )


@pytest.mark.parametrize('allow_invalid', [False, True])
def test_reject_invalid_kind_from_strategy(allow_invalid: bool) -> None:
    """Assert draws of a content that is not a kind of categories are rejected."""
    content = ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 3])), NumpyArray(np.array([1, 2, 3]))
    )
    with pytest.raises(Unsatisfiable):
        find(
            st_ak.contents.categorical_contents(
                st.just(content), allow_invalid=allow_invalid
            ),
            lambda _: True,
            settings=FIND_NO_SHRINK,
        )


def test_reject_non_unique_content_from_strategy() -> None:
    """Assert draws of non-unique categories from a strategy are rejected."""
    content = NumpyArray(np.array([1, 2, 1]))
    with pytest.raises(Unsatisfiable):
        find(
            st_ak.contents.categorical_contents(st.just(content), allow_invalid=False),
            lambda _: True,
            settings=FIND_NO_SHRINK,
        )


def _is_categorical(c: Content) -> bool:
    return c.parameter('__array__') == 'categorical'


def _n_nat(c: Content) -> int:
    if not isinstance(c, NumpyArray) or c.data.dtype.kind not in 'mM':
        return 0
    return int(np.isnat(c.data).sum())


def _n_nan(c: Content) -> int:
    if not isinstance(c, NumpyArray) or c.data.dtype.kind not in 'fc':
        return 0
    return int(np.isnan(c.data).sum())


def _has_duplicate_category(c: Content) -> bool:
    values = ak.Array(c).to_list()
    return any(values[i] == values[j] for j in range(len(values)) for i in range(j))


def _has_nat(c: Content) -> bool:
    """True if `c` contains a `NaT`."""
    return (
        isinstance(c, NumpyArray)
        and c.data.dtype.kind in 'mM'
        and bool(np.isnat(c.data).any())
    )


def _has_repeated_category(c: Content) -> bool:
    entries = [i for i in cast(Any, c).index.data.tolist() if i >= 0]
    return len(set(entries)) < len(entries)


def _has_unused_category(c: Content) -> bool:
    entries = {i for i in cast(Any, c).index.data.tolist() if i >= 0}
    return len(entries) < len(cast(Any, c).content)
