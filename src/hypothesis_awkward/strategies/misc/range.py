from typing import Any, Protocol, TypeVar, overload

from hypothesis import strategies as st

from hypothesis_awkward._util import safe_max, safe_min

from .idiom import none_or

_T = TypeVar('_T')


class _StMinMaxValuesFactory(Protocol[_T]):  # pragma: no cover
    def __call__(
        self, min_value: _T | None = None, max_value: _T | None = None
    ) -> st.SearchStrategy[_T]: ...


# For the default `st_=st.integers`, which mypy rejects for generic `_T`.
@overload
def ranges(
    *,
    min_start: int | None = None,
    max_start: int | None = None,
    min_end: int | None = None,
    max_end: int | None = None,
    allow_start_none: bool = True,
    allow_end_none: bool = True,
    let_end_none_if_start_none: bool = False,
    allow_equal: bool = True,
) -> st.SearchStrategy[tuple[int | None, int | None]]: ...


@overload
def ranges(
    st_: _StMinMaxValuesFactory[_T],
    /,
    *,
    min_start: _T | None = None,
    max_start: _T | None = None,
    min_end: _T | None = None,
    max_end: _T | None = None,
    allow_start_none: bool = True,
    allow_end_none: bool = True,
    let_end_none_if_start_none: bool = False,
    allow_equal: bool = True,
) -> st.SearchStrategy[tuple[_T | None, _T | None]]: ...


def ranges(
    st_: _StMinMaxValuesFactory[Any] = st.integers,
    /,
    *,
    min_start: Any | None = None,
    max_start: Any | None = None,
    min_end: Any | None = None,
    max_end: Any | None = None,
    allow_start_none: bool = True,
    allow_end_none: bool = True,
    let_end_none_if_start_none: bool = False,
    allow_equal: bool = True,
) -> st.SearchStrategy[tuple[Any | None, Any | None]]:
    """Generate two values (start, end) from a strategy, where start <= end.

    The minimum and maximum values can be specified by `min_start`,
    `max_start`, `min_end`, `max_end`.

    `start` (`end`) can be `None` if `allow_start_none` (`allow_end_none`) is `True`.

    If `let_end_none_if_start_none` is `True`, `end` will be always `None` when
    `start` is `None` regardless of `allow_end_none`.

    If `allow_equal` is `False`, `start` and `end` cannot be equal, i.e., `start < end`.

    Examples
    --------
    >>> from hypothesis import given
    >>> from hypothesis import strategies as st
    >>> import hypothesis_awkward.strategies as st_ak

    The pair can be passed to any strategy that takes a minimum and a maximum. Here,
    they are `min_length` and `max_length`:

    >>> @given(data=st.data())
    ... def test_lengths(data):
    ...     min_length, max_length = data.draw(
    ...         st_ak.misc.ranges(min_start=0, max_end=5, allow_start_none=False)
    ...     )
    ...     c = data.draw(
    ...         st_ak.constructors.list_offset_array_contents(
    ...             min_length=min_length, max_length=max_length
    ...         )
    ...     )
    ...     assert min_length <= len(c)
    ...     assert max_length is None or len(c) <= max_length
    >>> test_lengths()
    """

    def starts() -> st.SearchStrategy[Any | None]:
        _max_start = safe_min((max_start, max_end))
        _st = st_(min_value=min_start, max_value=_max_start)
        return none_or(_st) if allow_start_none else _st

    def ends(start: Any | None) -> st.SearchStrategy[Any | None]:
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
