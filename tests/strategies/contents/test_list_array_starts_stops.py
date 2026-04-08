from typing import TypedDict, cast

from hypothesis import find, given, settings
from hypothesis import strategies as st

from hypothesis_awkward.strategies.contents.list_array import _st_starts_stops
from hypothesis_awkward.util.safe import safe_compare as sc


class StartsStopsKwargs(TypedDict, total=False):
    """Options for `_st_starts_stops()`."""

    content_len: int
    max_length: int | None
    allow_unreachable: bool


@st.composite
def starts_stops_kwargs(draw: st.DrawFn) -> StartsStopsKwargs:
    """Strategy for options for `_st_starts_stops()`."""
    content_len = draw(st.integers(min_value=0, max_value=50))

    drawn = (('content_len', content_len),)

    kwargs = draw(
        st.fixed_dictionaries(
            {k: st.just(v) for k, v in drawn if v is not None},
            optional={
                'max_length': st.integers(min_value=0, max_value=content_len),
                'allow_unreachable': st.booleans(),
            },
        )
    )

    return cast(StartsStopsKwargs, kwargs)


@settings(max_examples=500)
@given(data=st.data())
def test_starts_stops(data: st.DataObject) -> None:
    """Test that `_st_starts_stops()` respects all its options."""
    # Draw options
    kwargs = data.draw(starts_stops_kwargs(), label='kwargs')

    # Call the test subject
    starts, stops = data.draw(_st_starts_stops(**kwargs), label='result')

    # Assert the options were effective
    content_len = kwargs['content_len']
    max_length = kwargs.get('max_length')
    allow_unreachable = kwargs.get('allow_unreachable', True)

    # Starts and stops have the same length and respect max_length
    assert len(starts) == len(stops) <= sc(max_length)

    # starts[i] <= stops[i] for all i
    for i in range(len(starts)):
        assert starts[i] <= stops[i]

    # All values are within [0, content_len]
    assert all(0 <= s <= content_len for s in starts)
    assert all(0 <= s <= content_len for s in stops)

    if not allow_unreachable:
        # No unreachable data: starts[0] == 0 and stops[-1] == content_len
        if len(starts) > 0:
            assert starts[0] == 0
            assert stops[-1] == content_len

    # Monotonic starts
    for i in range(len(starts) - 1):
        assert starts[i] <= starts[i + 1]

    if not allow_unreachable:
        # Contiguous: stops[i] == starts[i+1]
        for i in range(len(starts) - 1):
            assert stops[i] == starts[i + 1]


def test_draw_max_length() -> None:
    """Assert that starts/stops with exactly max_length lists can be drawn."""
    find(_st_starts_stops(5, max_length=10), lambda ss: len(ss[0]) == 10)


def test_draw_empty() -> None:
    """Assert that empty starts/stops can be drawn."""
    find(_st_starts_stops(5), lambda ss: len(ss[0]) == 0)


def test_draw_content_len_zero() -> None:
    """Assert that starts/stops can be drawn with zero content length."""
    starts, stops = find(
        _st_starts_stops(0, max_length=5),
        lambda ss: len(ss[0]) >= 2,
    )
    assert all(s == 0 for s in starts)
    assert all(s == 0 for s in stops)


def test_draw_unreachable_head() -> None:
    """Assert that starts/stops with unreachable head data can be drawn."""
    find(
        _st_starts_stops(10),
        lambda ss: len(ss[0]) >= 1 and ss[0][0] > 0,
    )


def test_draw_unreachable_tail() -> None:
    """Assert that starts/stops with unreachable tail data can be drawn."""
    find(
        _st_starts_stops(10),
        lambda ss: len(ss[0]) >= 1 and ss[1][-1] < 10,
    )


def test_shrink_no_unreachable() -> None:
    """Assert that starts/stops shrink toward no unreachable data."""
    starts, stops = find(
        _st_starts_stops(10),
        lambda ss: len(ss[0]) >= 2,
    )
    assert starts[0] == 0
    assert stops[-1] == 10


def test_draw_gap() -> None:
    """Assert that starts/stops with a gap between sublists can be drawn."""
    find(
        _st_starts_stops(10),
        lambda ss: any(ss[1][i] < ss[0][i + 1] for i in range(len(ss[0]) - 1)),
    )


def test_shrink_no_gap() -> None:
    """Assert that starts/stops shrink toward no gaps (contiguous)."""
    starts, stops = find(
        _st_starts_stops(10),
        lambda ss: len(ss[0]) >= 2,
    )
    for i in range(len(starts) - 1):
        assert stops[i] == starts[i + 1]


def test_draw_overlap() -> None:
    """Assert that starts/stops with overlapping sublists can be drawn."""
    find(
        _st_starts_stops(10),
        lambda ss: any(ss[0][i + 1] < ss[1][i] for i in range(len(ss[0]) - 1)),
    )


def test_shrink_no_overlap() -> None:
    """Assert that starts/stops shrink toward no overlaps."""
    starts, stops = find(
        _st_starts_stops(10),
        lambda ss: len(ss[0]) >= 2,
    )
    for i in range(len(starts) - 1):
        assert starts[i + 1] >= stops[i]


def test_shrink_content_len_zero() -> None:
    """Assert that starts/stops shrink to empty with zero content length."""
    starts, stops = find(
        _st_starts_stops(0),
        lambda ss: True,
    )
    assert starts == []
    assert stops == []
