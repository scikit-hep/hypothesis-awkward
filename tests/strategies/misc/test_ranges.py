import sys
from functools import partial
from typing import Any, TypeVar, cast

from hypothesis import given
from hypothesis import strategies as st

from hypothesis_awkward.strategies import StMinMaxValuesFactory, none_or, ranges
from hypothesis_awkward.util import safe_compare as sc
from hypothesis_awkward.util import safe_max
from tests.funcs import assert_kwargs_match_signature

if sys.version_info >= (3, 11):
    from typing import Generic, TypedDict
else:
    from typing_extensions import Generic, TypedDict

T = TypeVar('T')


def min_max_starts(
    st_: StMinMaxValuesFactory[T],
) -> st.SearchStrategy[tuple[T | None, T | None]]:
    """Strategy for an ordered pair of optional bounds on the start."""

    def mins() -> st.SearchStrategy[T | None]:
        return none_or(st_())

    def maxes(min_value: T | None) -> st.SearchStrategy[T | None]:
        return none_or(st_(min_value=min_value))

    return mins().flatmap(lambda min_: st.tuples(st.just(min_), maxes(min_)))


def min_max_ends(
    st_: StMinMaxValuesFactory[T],
    min_start: T | None = None,
) -> st.SearchStrategy[tuple[T | None, T | None]]:
    """Strategy for an ordered pair of optional bounds on the end.

    Both bounds are at least `min_start`.
    """

    def mins() -> st.SearchStrategy[T | None]:
        return none_or(st_(min_value=min_start))

    def maxes(min_value: T | None) -> st.SearchStrategy[T | None]:
        min_value = safe_max((min_value, min_start))
        return none_or(st_(min_value=min_value))

    return mins().flatmap(lambda min_: st.tuples(st.just(min_), maxes(min_)))


class RangesKwargs(TypedDict, Generic[T], total=False):
    """Options for `ranges()` strategy."""

    min_start: T | None
    max_start: T | None
    min_end: T | None
    max_end: T | None
    allow_start_none: bool
    allow_end_none: bool
    let_end_none_if_start_none: bool
    allow_equal: bool


DEFAULTS = RangesKwargs[Any](
    min_start=None,
    max_start=None,
    min_end=None,
    max_end=None,
    allow_start_none=True,
    allow_end_none=True,
    let_end_none_if_start_none=False,
    allow_equal=True,
)


class RelatedKwargs(TypedDict, Generic[T], total=False):
    """Options drawn together so the bounds are consistently ordered.

    `min_start <= max_start` and `min_start <= min_end <= max_end`.
    """

    min_start: T | None
    max_start: T | None
    min_end: T | None
    max_end: T | None


def test_kwargs_match_signature() -> None:
    """Assert the option declarations agree with `ranges()`'s parameters."""
    assert_kwargs_match_signature(
        func=ranges,
        exclude={'st_'},
        kwargs_cls=RangesKwargs,
        defaults=DEFAULTS,
        related_cls=RelatedKwargs,
    )


@st.composite
def ranges_kwargs(
    draw: st.DrawFn, st_: StMinMaxValuesFactory[T] | None = None
) -> RangesKwargs[T]:
    """Strategy for options for `ranges()` strategy.

    The bounds are drawn from `st_`, the factory the caller will pass to
    `ranges()`; `None` stands for the caller relying on `ranges()`'s default
    (`st.integers`).
    """
    if st_ is None:
        st_ = cast('StMinMaxValuesFactory[T]', st.integers)

    @st.composite
    def _st_related_kwargs(
        draw: st.DrawFn, st_: StMinMaxValuesFactory[T]
    ) -> RelatedKwargs[T]:
        """Strategy for the options that are drawn together."""
        min_start, max_start = draw(min_max_starts(st_=st_))  # type: ignore[arg-type]
        min_end, max_end = draw(
            min_max_ends(st_=st_, min_start=min_start)  # type: ignore[arg-type]
        )
        ret = RelatedKwargs[T]()
        if min_start is not None:
            ret['min_start'] = min_start
        if max_start is not None:
            ret['max_start'] = max_start
        if min_end is not None:
            ret['min_end'] = min_end
        if max_end is not None:
            ret['max_end'] = max_end
        return ret

    related = draw(_st_related_kwargs(st_))

    optional_independent = draw(
        st.fixed_dictionaries(
            {},
            optional={
                'allow_start_none': st.booleans(),
                'allow_end_none': st.booleans(),
                'let_end_none_if_start_none': st.booleans(),
                'allow_equal': st.booleans(),
            },
        )
    )

    return cast('RangesKwargs[T]', {**related, **optional_independent})


st_floats = partial(st.floats, allow_nan=False, allow_infinity=False)


@given(data=st.data())
def test_ranges_kwargs(data: st.DataObject) -> None:
    """Assert the invariants of the kwargs strategy itself."""
    st_ = data.draw(st.sampled_from([None, st_floats]))
    kwargs = data.draw(ranges_kwargs(st_=st_))  # type: ignore[arg-type]

    min_start = kwargs.get('min_start')
    max_start = kwargs.get('max_start')
    assert sc(min_start) <= sc(max_start)

    min_end = kwargs.get('min_end')
    max_end = kwargs.get('max_end')
    assert sc(min_start) <= sc(min_end) <= sc(max_end)


@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """Assert the results of `ranges()`."""
    # Draw options
    st_ = data.draw(st.sampled_from([None, st_floats]))
    kwargs = data.draw(ranges_kwargs(st_=st_))  # type: ignore[arg-type]

    args = (st_,) if st_ is not None else ()

    # Call the test subject
    start, end = data.draw(ranges(*args, **kwargs))  # type: ignore[arg-type]

    # Assert the options were effective
    allow_start_none = kwargs.get('allow_start_none', DEFAULTS['allow_start_none'])
    if not allow_start_none:
        assert start is not None

    let_end_none_if_start_none = kwargs.get(
        'let_end_none_if_start_none', DEFAULTS['let_end_none_if_start_none']
    )
    allow_end_none = kwargs.get('allow_end_none', DEFAULTS['allow_end_none'])
    if start is None and let_end_none_if_start_none:
        assert end is None
    elif not allow_end_none:
        assert end is not None

    allow_equal = kwargs.get('allow_equal', DEFAULTS['allow_equal'])
    if allow_equal:
        assert sc(start) <= sc(end)
    else:
        assert sc(start) < sc(end)

    min_start = kwargs.get('min_start')
    max_start = kwargs.get('max_start')
    assert sc(min_start) <= sc(start) <= sc(max_start)

    min_end = kwargs.get('min_end')
    max_end = kwargs.get('max_end')
    assert sc(min_end) <= sc(end) <= sc(max_end)
