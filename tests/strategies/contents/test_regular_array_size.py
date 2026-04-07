from typing import TypedDict, cast

from hypothesis import Phase, find, given, settings
from hypothesis import strategies as st

from hypothesis_awkward.strategies.contents.regular_array import _st_group_sizes


class GroupSizesKwargs(TypedDict, total=False):
    """Options for `_st_group_sizes()`."""

    total_items: int
    max_group_size: int
    max_length: int | None


@st.composite
def group_sizes_kwargs(draw: st.DrawFn) -> GroupSizesKwargs:
    """Strategy for options for `_st_group_sizes()`."""
    total_items = draw(st.integers(min_value=0, max_value=100))
    max_group_size = draw(st.integers(min_value=0, max_value=100))

    drawn = (
        ('total_items', total_items),
        ('max_group_size', max_group_size),
    )

    kwargs = draw(
        st.fixed_dictionaries(
            {k: st.just(v) for k, v in drawn if v is not None},
            optional={
                'max_length': st.integers(min_value=0, max_value=100),
            },
        )
    )

    return cast(GroupSizesKwargs, kwargs)


@settings(max_examples=500)
@given(data=st.data())
def test_group_sizes(data: st.DataObject) -> None:
    """Test that `_st_group_sizes()` respects all its options."""
    # Draw options
    kwargs = data.draw(group_sizes_kwargs(), label='kwargs')

    # Call the test subject
    result = data.draw(_st_group_sizes(**kwargs), label='result')

    # Assert the options were effective
    total_items = kwargs['total_items']
    max_group_size = kwargs['max_group_size']
    max_length = kwargs.get('max_length')

    # Result is bounded by max_group_size
    assert 0 <= result <= max_group_size

    # Result is bounded by total_items when total_items > 0
    if total_items > 0:
        assert result <= total_items

    # Result divides total_items when both are positive
    if total_items > 0 and result > 0:
        assert total_items % result == 0

    # max_length is respected when result > 0
    if max_length is not None and result > 0:
        assert total_items // result <= max_length


def test_shrink_to_one() -> None:
    """Assert that positive size shrinks to 1."""
    s = find(_st_group_sizes(total_items=12, max_group_size=12), lambda s: s > 0)
    assert s == 1


def test_draw_total_items() -> None:
    """Assert that size can equal total_items."""
    find(
        _st_group_sizes(total_items=7, max_group_size=7),
        lambda s: s == 7,
        settings=settings(phases=[Phase.generate]),
    )


def test_draw_total_items_zero() -> None:
    """Assert that any size up to max_group_size can be drawn when total_items is 0."""
    max_group_size = 10
    find(
        _st_group_sizes(total_items=0, max_group_size=max_group_size),
        lambda s: s == max_group_size,
        settings=settings(phases=[Phase.generate]),
    )
