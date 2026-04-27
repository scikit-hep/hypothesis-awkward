from typing import TypedDict, cast

from hypothesis import find, given, settings
from hypothesis import strategies as st

from hypothesis_awkward import strategies as st_ak
from hypothesis_awkward.strategies.contents.regular_array import _st_group_sizes
from hypothesis_awkward.util import safe_compare as sc


class GroupSizesKwargs(TypedDict, total=False):
    """Options for `_st_group_sizes()`."""

    total_items: int
    min_group_size: int
    max_group_size: int | None
    min_length: int
    max_length: int | None
    allow_non_divisors: bool


@st.composite
def group_sizes_kwargs(draw: st.DrawFn) -> GroupSizesKwargs:
    """Strategy for options for `_st_group_sizes()`."""
    total_items = draw(st.integers(min_value=0, max_value=100))
    min_group_size, max_group_size = draw(
        st_ak.ranges(min_start=0, max_end=total_items)
    )
    min_length, max_length = draw(st_ak.ranges(min_start=0, max_end=total_items))

    drawn = (
        ('total_items', total_items),
        ('min_group_size', min_group_size),
        ('max_group_size', max_group_size),
        ('min_length', min_length),
        ('max_length', max_length),
    )

    kwargs = draw(
        st.fixed_dictionaries(
            {k: st.just(v) for k, v in drawn if v is not None},
            optional={
                'allow_non_divisors': st.booleans(),
            },
        )
    )

    return cast(GroupSizesKwargs, kwargs)


@settings(max_examples=500)
@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """Assert the results of `_st_group_sizes()`."""
    # Draw options
    kwargs = data.draw(group_sizes_kwargs(), label='kwargs')

    # Call the test subject
    result = data.draw(_st_group_sizes(**kwargs), label='result')

    # Assert the options were effective
    total_items = kwargs['total_items']
    min_group_size = kwargs.get('min_group_size', 0)
    max_group_size = kwargs.get('max_group_size')
    min_length = kwargs.get('min_length', 0)
    max_length = kwargs.get('max_length')
    allow_non_divisors = kwargs.get('allow_non_divisors', True)

    # Result is bounded by min/max_group_size
    assert result >= 0
    if max_group_size is not None:
        assert result <= max_group_size

    # Result respects min_group_size (except fallback to 0)
    if result > 0:
        assert result >= min_group_size

    # Result is bounded by total_items when total_items > 0
    if total_items > 0:
        assert result <= total_items

    # Result divides total_items when both are positive and non-divisors disallowed
    if total_items > 0 and result > 0 and not allow_non_divisors:
        assert total_items % result == 0

    # min_length / max_length are respected when result > 0;
    # result == 0 defers length enforcement to the caller's size==0 branch.
    if result > 0:
        assert min_length <= total_items // result <= sc(max_length)


def test_shrink_to_total_items() -> None:
    """Assert that positive size shrinks to total_items (fewest partitions)."""
    s = find(
        _st_group_sizes(12, max_group_size=12),
        lambda s: s > 0,
        settings=settings(database=None),
    )
    assert s == 12


def test_shrink_to_max_divisor() -> None:
    """Assert that positive size shrinks to the largest divisor <= max_group_size."""
    # total_items=12, max_group_size=11: divisors <= 11 are [6, 4, 3, 2, 1]
    s = find(
        _st_group_sizes(12, max_group_size=11),
        lambda s: s > 0,
        settings=settings(database=None),
    )
    assert s == 6


def test_draw_total_items() -> None:
    """Assert that size can equal total_items."""
    find(_st_group_sizes(7, max_group_size=7), lambda s: s == 7)


def test_draw_total_items_zero() -> None:
    """Assert that any size up to max_group_size can be drawn when total_items is 0."""
    max_group_size = 10
    find(
        _st_group_sizes(0, max_group_size=max_group_size), lambda s: s == max_group_size
    )


def test_shrink_divisors_first() -> None:
    """Assert that with allow_non_divisors, shrinking prefers divisors."""
    # total_items=12, max_group_size=11: divisors <= 11 are [6, 4, 3, 2, 1],
    # non-divisors <= 11 are [11, 10, 9, 8, 7, 5].
    # Shrink should pick 6 (largest divisor), not 11 (largest non-divisor).
    s = find(
        _st_group_sizes(12, max_group_size=11, allow_non_divisors=True),
        lambda s: s > 0,
        settings=settings(database=None),
    )
    assert s == 6


def test_draw_non_divisor() -> None:
    """Assert that a non-divisor can be drawn when allow_non_divisors is True."""
    find(_st_group_sizes(12, allow_non_divisors=True), lambda s: s > 0 and 12 % s != 0)


def test_min_length_caps_size() -> None:
    """Assert that min_length caps the resolved max_group_size."""
    # total_items=12, min_length=3 → size <= 12 // 3 = 4.
    # Largest divisor of 12 in [1, 4] is 4.
    s = find(
        _st_group_sizes(12, min_length=3),
        lambda s: s > 0,
        settings=settings(database=None),
    )
    assert s == 4


def test_draw_min_length_zero_fallback() -> None:
    """Assert size=0 fallback when `min_length > total_items`."""
    find(_st_group_sizes(3, min_length=5), lambda s: s == 0)


def test_draw_min_length_total_items_zero() -> None:
    """Assert size=0 fallback when total_items=0 and min_length>0."""
    find(_st_group_sizes(0, min_length=1), lambda s: s == 0)
