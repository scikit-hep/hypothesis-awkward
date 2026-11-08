from typing import TypedDict, cast

from hypothesis import find, given
from hypothesis import strategies as st

from hypothesis_awkward import strategies as st_ak
from hypothesis_awkward._util import safe_compare as sc
from hypothesis_awkward.strategies.constructors.list_array import _st_starts_stops
from tests.find_settings import FIND
from tests.funcs import assert_kwargs_match_signature


class StartsStopsKwargs(TypedDict, total=False):
    """Options for `_st_starts_stops()`."""

    min_length: int
    max_length: int | None
    allow_unreachable: bool


DEFAULTS = StartsStopsKwargs(min_length=0, max_length=None, allow_unreachable=True)


class RelatedKwargs(TypedDict, total=False):
    """Options drawn together (`min_length <= max_length <= content_len`)."""

    min_length: int
    max_length: int | None


def test_kwargs_match_signature() -> None:
    """Assert the option declarations agree with `_st_starts_stops()`."""
    assert_kwargs_match_signature(
        func=_st_starts_stops,
        exclude={'content_len'},
        kwargs_cls=StartsStopsKwargs,
        defaults=DEFAULTS,
        related_cls=RelatedKwargs,
    )


@st.composite
def starts_stops_kwargs(draw: st.DrawFn, content_len: int) -> StartsStopsKwargs:
    """Strategy for options for `_st_starts_stops()`.

    The length bounds are drawn within `content_len`, the positional argument the
    caller passes to `_st_starts_stops()`.
    """

    @st.composite
    def _st_related_kwargs(draw: st.DrawFn) -> RelatedKwargs:
        """Strategy for the options that are drawn together."""
        min_length, max_length = draw(
            st_ak.misc.ranges(min_start=0, max_end=content_len)
        )
        ret = RelatedKwargs()
        if min_length is not None:
            ret['min_length'] = min_length
        if max_length is not None:
            ret['max_length'] = max_length
        return ret

    related = draw(_st_related_kwargs())

    optional_independent = draw(
        st.fixed_dictionaries(
            {},
            optional={
                'allow_unreachable': st.booleans(),
            },
        )
    )

    return cast(StartsStopsKwargs, {**related, **optional_independent})


@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """Assert the results of `_st_starts_stops()`."""
    # Draw options
    content_len = data.draw(st.integers(min_value=0, max_value=50), label='content_len')
    kwargs = data.draw(starts_stops_kwargs(content_len=content_len), label='kwargs')

    # Call the test subject
    starts, stops = data.draw(_st_starts_stops(content_len, **kwargs), label='result')

    # Assert the options were effective
    min_length = kwargs.get('min_length', DEFAULTS['min_length'])
    max_length = kwargs.get('max_length', DEFAULTS['max_length'])
    allow_unreachable = kwargs.get('allow_unreachable', DEFAULTS['allow_unreachable'])

    # Starts and stops have the same length and respect min_length / max_length
    assert min_length <= len(starts) == len(stops) <= sc(max_length)

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

    if not allow_unreachable:
        # Monotonic starts
        for i in range(len(starts) - 1):
            assert starts[i] <= starts[i + 1]

    if not allow_unreachable:
        # Contiguous: stops[i] == starts[i+1]
        for i in range(len(starts) - 1):
            assert stops[i] == starts[i + 1]


def test_draw_min_length() -> None:
    """Assert that starts/stops with exactly min_length lists can be drawn."""
    find(_st_starts_stops(5, min_length=3), lambda ss: len(ss[0]) == 3, settings=FIND)


def test_draw_min_length_content_len_zero() -> None:
    """Assert that min_length is respected when content is empty."""
    find(_st_starts_stops(0, min_length=3), lambda ss: len(ss[0]) == 3, settings=FIND)


def test_draw_max_length() -> None:
    """Assert that starts/stops with exactly max_length lists can be drawn."""
    find(_st_starts_stops(5, max_length=10), lambda ss: len(ss[0]) == 10, settings=FIND)


def test_draw_empty() -> None:
    """Assert that empty starts/stops can be drawn."""
    find(_st_starts_stops(5), lambda ss: len(ss[0]) == 0, settings=FIND)


def test_draw_content_len_zero() -> None:
    """Assert that starts/stops can be drawn with zero content length."""
    starts, stops = find(
        _st_starts_stops(0, max_length=5), lambda ss: len(ss[0]) >= 2, settings=FIND
    )
    assert all(s == 0 for s in starts)
    assert all(s == 0 for s in stops)


def test_draw_unreachable_head() -> None:
    """Assert that starts/stops with unreachable head data can be drawn."""
    find(
        _st_starts_stops(10), lambda ss: len(ss[0]) >= 1 and ss[0][0] > 0, settings=FIND
    )


def test_draw_unreachable_tail() -> None:
    """Assert that starts/stops with unreachable tail data can be drawn."""
    find(
        _st_starts_stops(10),
        lambda ss: len(ss[0]) >= 1 and ss[1][-1] < 10,
        settings=FIND,
    )


def test_shrink_no_unreachable() -> None:
    """Assert that starts/stops shrink toward no unreachable data."""
    starts, stops = find(
        _st_starts_stops(10), lambda ss: len(ss[0]) >= 2, settings=FIND
    )
    assert starts[0] == 0
    assert stops[-1] == 10


def test_draw_gap() -> None:
    """Assert that starts/stops with a gap between sublists can be drawn."""
    find(
        _st_starts_stops(10),
        lambda ss: any(ss[1][i] < ss[0][i + 1] for i in range(len(ss[0]) - 1)),
        settings=FIND,
    )


def test_shrink_no_gap() -> None:
    """Assert that starts/stops shrink toward no gaps (contiguous)."""
    starts, stops = find(
        _st_starts_stops(10), lambda ss: len(ss[0]) >= 2, settings=FIND
    )
    for i in range(len(starts) - 1):
        assert stops[i] == starts[i + 1]


def test_draw_overlap() -> None:
    """Assert that starts/stops with overlapping sublists can be drawn."""
    find(
        _st_starts_stops(10),
        lambda ss: any(ss[0][i + 1] < ss[1][i] for i in range(len(ss[0]) - 1)),
        settings=FIND,
    )


def test_shrink_no_overlap() -> None:
    """Assert that starts/stops shrink toward no overlaps."""
    starts, stops = find(
        _st_starts_stops(10), lambda ss: len(ss[0]) >= 2, settings=FIND
    )
    for i in range(len(starts) - 1):
        assert starts[i + 1] >= stops[i]


def test_draw_out_of_order() -> None:
    """Assert that starts/stops with out-of-order starts can be drawn."""
    find(
        _st_starts_stops(10),
        lambda ss: any(ss[0][i] > ss[0][i + 1] for i in range(len(ss[0]) - 1)),
        settings=FIND,
    )


def test_shrink_monotonic() -> None:
    """Assert that starts/stops shrink toward monotonic starts."""
    starts, stops = find(
        _st_starts_stops(10), lambda ss: len(ss[0]) >= 2, settings=FIND
    )
    for i in range(len(starts) - 1):
        assert starts[i] <= starts[i + 1]


def test_shrink_content_len_zero() -> None:
    """Assert that starts/stops shrink to empty with zero content length."""
    starts, stops = find(_st_starts_stops(0), lambda ss: True, settings=FIND)
    assert starts == []
    assert stops == []
