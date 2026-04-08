from typing import TypedDict, cast

from hypothesis import find, given, settings
from hypothesis import strategies as st

from hypothesis_awkward.strategies.contents.list_array import _st_starts_stops
from hypothesis_awkward.util.safe import safe_compare as sc


class StartsStopsKwargs(TypedDict, total=False):
    """Options for `_st_starts_stops()`."""

    content_len: int
    max_length: int | None


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

    # Starts and stops have the same length and respect max_length
    assert len(starts) == len(stops) <= sc(max_length)

    # starts[i] <= stops[i] for all i
    for i in range(len(starts)):
        assert starts[i] <= stops[i]

    # All values are within [0, content_len]
    assert all(0 <= s <= content_len for s in starts)
    assert all(0 <= s <= content_len for s in stops)

    # starts[0] == 0 and stops[-1] == content_len (no unreachable data)
    if len(starts) > 0:
        assert starts[0] == 0
        assert stops[-1] == content_len

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


def test_shrink_content_len_zero() -> None:
    """Assert that starts/stops shrink to empty with zero content length."""
    starts, stops = find(
        _st_starts_stops(0),
        lambda ss: True,
    )
    assert starts == []
    assert stops == []
