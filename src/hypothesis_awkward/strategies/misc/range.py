from typing import Any, Optional, Protocol, TypeVar, overload

from hypothesis import strategies as st

from hypothesis_awkward._util import safe_max, safe_min

from .idiom import none_or

_T = TypeVar('_T')


class _StMinMaxValuesFactory(Protocol[_T]):  # pragma: no cover
    def __call__(
        self, min_value: Optional[_T] = None, max_value: Optional[_T] = None
    ) -> st.SearchStrategy[_T]: ...


# For the default `st_=st.integers`, which mypy rejects for generic `_T`.
@overload
def ranges(
    *,
    min_start: Optional[int] = None,
    max_start: Optional[int] = None,
    min_end: Optional[int] = None,
    max_end: Optional[int] = None,
    allow_start_none: bool = True,
    allow_end_none: bool = True,
    let_end_none_if_start_none: bool = False,
    allow_equal: bool = True,
) -> st.SearchStrategy[tuple[Optional[int], Optional[int]]]: ...


@overload
def ranges(
    st_: _StMinMaxValuesFactory[_T],
    /,
    *,
    min_start: Optional[_T] = None,
    max_start: Optional[_T] = None,
    min_end: Optional[_T] = None,
    max_end: Optional[_T] = None,
    allow_start_none: bool = True,
    allow_end_none: bool = True,
    let_end_none_if_start_none: bool = False,
    allow_equal: bool = True,
) -> st.SearchStrategy[tuple[Optional[_T], Optional[_T]]]: ...


def ranges(
    st_: _StMinMaxValuesFactory[Any] = st.integers,
    /,
    *,
    min_start: Optional[Any] = None,
    max_start: Optional[Any] = None,
    min_end: Optional[Any] = None,
    max_end: Optional[Any] = None,
    allow_start_none: bool = True,
    allow_end_none: bool = True,
    let_end_none_if_start_none: bool = False,
    allow_equal: bool = True,
) -> st.SearchStrategy[tuple[Optional[Any], Optional[Any]]]:
    """Generate two values (start, end) from a strategy, where start <= end.

    The minimum and maximum values can be specified by `min_start`,
    `max_start`, `min_end`, `max_end`.

    `start` (`end`) can be `None` if `allow_start_none` (`allow_end_none`) is `True`.

    If `let_end_none_if_start_none` is `True`, `end` will be always `None` when
    `start` is `None` regardless of `allow_end_none`.

    If `allow_equal` is `False`, `start` and `end` cannot be equal, i.e., `start < end`.

    >>> start, end = ranges(
    ...     st.integers,
    ...     min_start=0,
    ...     max_end=10,
    ...     allow_start_none=False,
    ...     allow_end_none=False,
    ... ).example()

    The results can be, for example, used as min_value and max_value of st.integers().

    >>> i = st.integers(min_value=start, max_value=end).example()
    >>> start <= i <= end
    True

    The results can also be used as min_size and max_size of st.lists().

    >>> l = st.lists(st.integers(), min_size=start, max_size=end).example()
    >>> start <= len(l) <= end
    True
    """

    def starts() -> st.SearchStrategy[Optional[Any]]:
        _max_start = safe_min((max_start, max_end))
        _st = st_(min_value=min_start, max_value=_max_start)
        return none_or(_st) if allow_start_none else _st

    def ends(start: Any | None) -> st.SearchStrategy[Optional[Any]]:
        _min_end = safe_max((min_start, start, min_end))
        if min_end is not None and max_end is not None:
            assert min_end <= max_end
        if start is None and let_end_none_if_start_none:
            return st.none()
        _st = st_(min_value=_min_end, max_value=max_end)
        if start is not None and not allow_equal:
            _st = _st.filter(lambda x: x > start)
        return none_or(_st) if allow_end_none else _st

    return starts().flatmap(lambda start: st.tuples(st.just(start), ends(start=start)))
