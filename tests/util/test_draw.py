from typing import TypedDict, cast

from hypothesis import HealthCheck, Phase, find, given, settings
from hypothesis import strategies as st

from hypothesis_awkward.util.draw import CountdownDrawer


def _sized_lists(*, max_size: int) -> st.SearchStrategy[list[int]]:
    '''A `_StWithMaxSize`-conforming callable for testing.'''
    return st.lists(st.integers(), max_size=max_size)


class CountdownDrawerKwargs(TypedDict):
    '''Options for `CountdownDrawer()`.'''

    max_size: int


def countdown_drawer_kwargs() -> st.SearchStrategy[CountdownDrawerKwargs]:
    '''Strategy for options for `CountdownDrawer()`.'''
    return st.fixed_dictionaries(
        {'max_size': st.integers(min_value=0, max_value=50)},
    ).map(lambda d: cast(CountdownDrawerKwargs, d))


@st.composite
def _exhaust(
    draw: st.DrawFn, kwargs: CountdownDrawerKwargs
) -> tuple[int, list[list[int]]]:
    '''Draw from a `CountdownDrawer` until it returns `None`.'''
    draw_content = CountdownDrawer(draw, _sized_lists, **kwargs)

    total = 0
    results: list[list[int]] = []

    while True:
        result = draw_content()
        if result is None:
            break
        results.append(result)
        total += len(result)

    # Once None is returned, subsequent calls also return None
    assert draw_content() is None
    assert draw_content() is None

    return total, results


@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(data=st.data())
def test_countdown_drawer(data: st.DataObject) -> None:
    kwargs = data.draw(countdown_drawer_kwargs(), label='kwargs')
    total, _ = data.draw(_exhaust(kwargs), label='result')

    # Total length of all returned lists is at most max_size
    assert total <= kwargs['max_size']


def test_draw_none_at_zero() -> None:
    '''Assert that CountdownDrawer with max_size=0 returns None on the first call.'''

    @st.composite
    def _scenario(draw: st.DrawFn) -> None:
        draw_content = CountdownDrawer(draw, _sized_lists, max_size=0)
        assert draw_content() is None

    find(
        _scenario(),
        lambda _: True,
        settings=settings(phases=[Phase.generate]),
    )


def test_draw_max_size() -> None:
    '''Assert that the total can reach max_size.'''
    max_size = 10

    @st.composite
    def _scenario(draw: st.DrawFn) -> int:
        draw_content = CountdownDrawer(draw, _sized_lists, max_size=max_size)
        total = 0
        while True:
            result = draw_content()
            if result is None:
                break
            total += len(result)
        return total

    find(
        _scenario(),
        lambda total: total == max_size,
        settings=settings(phases=[Phase.generate]),
    )


def test_draw_multiple() -> None:
    '''Assert that a CountdownDrawer can produce more than one non-None result.'''

    @st.composite
    def _scenario(draw: st.DrawFn) -> int:
        draw_content = CountdownDrawer(draw, _sized_lists, max_size=50)
        count = 0
        while True:
            result = draw_content()
            if result is None:
                break
            count += 1
        return count

    find(
        _scenario(),
        lambda count: count > 1,
        settings=settings(phases=[Phase.generate]),
    )
