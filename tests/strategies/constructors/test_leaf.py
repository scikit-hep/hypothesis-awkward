from typing import Any, TypedDict, cast

import numpy as np
import pytest
from hypothesis import find, given
from hypothesis import strategies as st

import awkward as ak
from awkward.contents import EmptyArray, NumpyArray
from hypothesis_awkward import strategies as st_ak
from hypothesis_awkward._util import (
    any_nan_in_awkward_array,
    any_nan_nat_in_numpy_array,
    is_bytestring_leaf,
    is_string_leaf,
    is_zero_field_record_leaf,
)
from hypothesis_awkward._util import safe_compare as sc
from tests.find_settings import FIND, FIND_RARE
from tests.funcs import KwargsDrawn, all_distinct, assert_kwargs_match_signature


class LeafContentsKwargs(TypedDict, total=False):
    """Options for `leaf_contents()` strategy."""

    dtypes: st.SearchStrategy[np.dtype] | None
    allow_nan: bool
    min_size: int
    max_size: int
    unique: bool
    allow_numpy: bool
    allow_empty: bool
    allow_string: bool
    allow_bytestring: bool
    allow_zero_field_record: bool


DEFAULTS = LeafContentsKwargs(
    dtypes=None,
    allow_nan=True,
    min_size=0,
    max_size=10,
    unique=False,
    allow_numpy=True,
    allow_empty=True,
    allow_string=True,
    allow_bytestring=True,
    allow_zero_field_record=True,
)

ALLOW_FLAGS = (
    'allow_numpy',
    'allow_empty',
    'allow_string',
    'allow_bytestring',
    'allow_zero_field_record',
)
NON_EMPTY_FLAGS = tuple(f for f in ALLOW_FLAGS if f != 'allow_empty')
# Leaf types that can hold more than one distinct element.
DISTINCT_FLAGS = tuple(f for f in NON_EMPTY_FLAGS if f != 'allow_zero_field_record')


class LeafContentsDrawn(TypedDict, total=False):
    """Values drawn in advance for the options of `leaf_contents()`."""

    dtypes: np.dtype


class RelatedKwargs(TypedDict, total=False):
    """Options drawn together.

    `min_size <= max_size`, and at least one leaf type is allowed that can have a
    length in that range (`EmptyArray` only when `min_size` is 0, a zero-field record
    under `unique` only when `min_size` is at most 1). They depend on the dtype drawn
    in advance: under `unique`, a `bool` dtype limits `min_size` to the two distinct
    `bool` values.
    """

    dtypes: st.SearchStrategy[np.dtype] | None
    min_size: int
    max_size: int
    unique: bool
    allow_numpy: bool
    allow_empty: bool
    allow_string: bool
    allow_bytestring: bool
    allow_zero_field_record: bool


def test_kwargs_match_signature() -> None:
    """Assert the option declarations agree with `leaf_contents()`."""
    assert_kwargs_match_signature(
        func=st_ak.constructors.leaf_contents,
        kwargs_cls=LeafContentsKwargs,
        defaults=DEFAULTS,
        related_cls=RelatedKwargs,
    )


@st.composite
def leaf_contents_kwargs(
    draw: st.DrawFn,
) -> KwargsDrawn[LeafContentsKwargs, LeafContentsDrawn]:
    """Strategy for options for `leaf_contents()` strategy."""

    @st.composite
    def _st_drawn(draw: st.DrawFn) -> LeafContentsDrawn:
        """Strategy for the values drawn in advance for the options."""
        ret = LeafContentsDrawn()
        if draw(st.booleans()):
            ret['dtypes'] = draw(st_ak.numpy.supported_dtypes())
        return ret

    @st.composite
    def _st_related_kwargs(draw: st.DrawFn, drawn: LeafContentsDrawn) -> RelatedKwargs:
        """Strategy for the options that are drawn together."""
        unique = draw(st_ak.misc.none_or(st.booleans()))
        max_start = DEFAULTS['max_size']
        if unique and 'dtypes' in drawn and drawn['dtypes'].kind == 'b':
            max_start = 2  # the number of distinct `bool` values
        min_size, max_size = draw(
            st_ak.misc.ranges(
                min_start=0,
                max_start=max_start,
                max_end=DEFAULTS['max_size'] + 50,
            )
        )
        ret = dict[str, Any]()
        if 'dtypes' in drawn:
            ret['dtypes'] = st.just(drawn['dtypes'])
        elif draw(st.booleans()):
            ret['dtypes'] = None
        if min_size is not None:
            ret['min_size'] = min_size
        if max_size is not None:
            ret['max_size'] = max_size
        if unique is not None:
            ret['unique'] = unique
        if unique and (min_size or 0) > 1:
            # A zero-field record, whose elements are all equal, is impossible too.
            feasible = DISTINCT_FLAGS
        elif min_size:
            # `EmptyArray` is impossible; at least one non-empty type is needed.
            feasible = NON_EMPTY_FLAGS
        else:
            feasible = ALLOW_FLAGS
        allowed = draw(st.sets(st.sampled_from(feasible), min_size=1))
        infeasible = [f for f in ALLOW_FLAGS if f not in feasible]
        if infeasible:
            # The impossible types may be allowed as well.
            allowed |= draw(st.sets(st.sampled_from(infeasible)))
        for flag in ALLOW_FLAGS:
            # A disallowed flag must be passed; an allowed one may be omitted.
            if flag not in allowed or draw(st.booleans()):
                ret[flag] = flag in allowed
        return cast(RelatedKwargs, ret)

    drawn = draw(_st_drawn())
    related = draw(_st_related_kwargs(drawn))
    optional_independent = cast(
        LeafContentsKwargs,
        draw(st.fixed_dictionaries({}, optional={'allow_nan': st.booleans()})),
    )

    kwargs = LeafContentsKwargs(**related, **optional_independent)
    return KwargsDrawn(kwargs, drawn)


@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """Assert the results of `leaf_contents()`."""
    # Draw options
    kwargs, drawn = data.draw(leaf_contents_kwargs(), label='opts')

    min_size = kwargs.get('min_size', DEFAULTS['min_size'])
    max_size = kwargs.get('max_size', DEFAULTS['max_size'])
    unique = kwargs.get('unique', DEFAULTS['unique'])
    allow_numpy = kwargs.get('allow_numpy', DEFAULTS['allow_numpy'])
    allow_empty = kwargs.get('allow_empty', DEFAULTS['allow_empty'])
    allow_string = kwargs.get('allow_string', DEFAULTS['allow_string'])
    allow_bytestring = kwargs.get('allow_bytestring', DEFAULTS['allow_bytestring'])
    allow_zero_field_record = kwargs.get(
        'allow_zero_field_record', DEFAULTS['allow_zero_field_record']
    )

    # Call the test subject
    result = data.draw(st_ak.constructors.leaf_contents(**kwargs), label='result')

    is_numpy = isinstance(result, NumpyArray)
    is_empty = isinstance(result, EmptyArray)
    is_string = is_string_leaf(result)
    is_bytestring = is_bytestring_leaf(result)
    is_zero_field_record = is_zero_field_record_leaf(result)

    assert any((is_numpy, is_empty, is_string, is_bytestring, is_zero_field_record))

    # Assert the options were effective
    if not allow_numpy:
        assert not is_numpy
    if not allow_empty:
        assert not is_empty
    if not allow_string:
        assert not is_string
    if not allow_bytestring:
        assert not is_bytestring
    if not allow_zero_field_record:
        assert not is_zero_field_record

    allow_nan = kwargs.get('allow_nan', DEFAULTS['allow_nan'])

    if is_empty:
        assert len(result) == 0
    else:
        assert sc(min_size) <= len(result) <= sc(max_size)

    if is_numpy:
        assert isinstance(result, NumpyArray)
        if not allow_nan:
            assert not any_nan_nat_in_numpy_array(result.data)
        if 'dtypes' in drawn:
            assert result.data.dtype == drawn['dtypes']

    if unique:
        match result:
            case NumpyArray():
                assert all_distinct(list(result.data))
            case _:
                assert all_distinct(ak.Array(result).to_list())


def test_draw_numpy_array() -> None:
    """Assert that NumpyArray can be drawn by default."""
    find(
        st_ak.constructors.leaf_contents(),
        lambda c: isinstance(c, NumpyArray),
        settings=FIND,
    )


def test_draw_empty_array() -> None:
    """Assert that EmptyArray can be drawn by default."""
    find(
        st_ak.constructors.leaf_contents(),
        lambda c: isinstance(c, EmptyArray),
        settings=FIND,
    )


def test_draw_string() -> None:
    """Assert that string content can be drawn by default."""
    find(st_ak.constructors.leaf_contents(), lambda c: is_string_leaf(c), settings=FIND)


def test_draw_bytestring() -> None:
    """Assert that bytestring content can be drawn by default."""
    find(
        st_ak.constructors.leaf_contents(),
        lambda c: is_bytestring_leaf(c),
        settings=FIND,
    )


@pytest.mark.parametrize('is_tuple', [True, False])
def test_draw_zero_field_record(is_tuple: bool) -> None:
    """Assert a zero-field record can be drawn by default."""
    find(
        st_ak.constructors.leaf_contents(),
        lambda c: (
            is_zero_field_record_leaf(c) and c.is_tuple == is_tuple and len(c) != 0
        ),
        settings=FIND,
    )


def test_draw_max_size() -> None:
    """Assert that leaf content with max_size elements can be drawn."""
    max_size = 25
    find(
        st_ak.constructors.leaf_contents(max_size=max_size),
        lambda c: len(c) == max_size,
        settings=FIND,
    )


def test_draw_unique_zero_field_record() -> None:
    """Assert a one-element zero-field record can be drawn when unique is True."""
    find(
        st_ak.constructors.leaf_contents(unique=True),
        lambda c: is_zero_field_record_leaf(c) and len(c) == 1,
        settings=FIND,
    )


def test_draw_nan() -> None:
    """Assert that leaf content with NaN can be drawn."""
    c = find(
        st_ak.constructors.leaf_contents(), any_nan_in_awkward_array, settings=FIND_RARE
    )
    assert isinstance(c, NumpyArray)
    assert np.array_equal(c.data, np.array([np.nan]), equal_nan=True)


def test_shrink_leaf_contents_empty() -> None:
    """Assert that leaf content can shrink to an empty array when allowed."""
    c = find(st_ak.constructors.leaf_contents(), lambda c: len(c) == 0, settings=FIND)
    assert isinstance(c, EmptyArray)


@pytest.mark.xfail(
    reason=(
        'bimodal shrink: the endpoint is [b\'\'] or [{}] depending on which '
        'one_of branch the first found example lands in'
    )
)
def test_shrink_leaf_contents_one() -> None:
    """Assert that leaf content with one element shrinks to [b'']."""
    # A zero-field record found first stays a record: the shrinker cannot
    # leave its one_of branch (EmptyArray fails len == 1), nor enter it from
    # another branch (the length draw falls back to 0). TODO: Revisit.
    c = find(st_ak.constructors.leaf_contents(), lambda c: len(c) == 1, settings=FIND)
    assert c.to_list() == [b'']


@pytest.mark.parametrize(
    'kwargs',
    [
        pytest.param({flag: False for flag in ALLOW_FLAGS}, id='all-disallowed'),
        pytest.param(
            {'min_size': 1, **{flag: False for flag in NON_EMPTY_FLAGS}},
            id='only-empty-with-min-size',
        ),
        pytest.param(
            {
                'min_size': 2,
                'unique': True,
                **{flag: False for flag in DISTINCT_FLAGS},
            },
            id='only-empty-and-zero-field-record-with-unique',
        ),
    ],
)
def test_raise_no_possible_type(kwargs: LeafContentsKwargs) -> None:
    """Assert `ValueError` when no allowed leaf type can satisfy the options."""
    with pytest.raises(
        ValueError, match='no content types are possible with the given options'
    ):
        st_ak.constructors.leaf_contents(**kwargs)
