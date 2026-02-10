from collections.abc import Callable, Sized
from typing import Protocol, TypeVar

from hypothesis import strategies as st

_C_co = TypeVar('_C_co', covariant=True)
_T = TypeVar('_T', bound=Sized)


class _StWithMaxSize(Protocol[_C_co]):
    '''A callable that takes a ``max_size`` keyword and returns a content strategy.'''

    def __call__(self, *, max_size: int) -> st.SearchStrategy[_C_co]: ...


def CountdownDrawer(
    draw: st.DrawFn,
    st_: _StWithMaxSize[_T],
    max_size: int,
) -> Callable[[], _T | None]:
    '''Create a draw function that counts down from ``max_size``.

    Each call draws from ``st_`` and subtracts the length of the result
    from the remaining count. Returns ``None`` once the count reaches zero.

    Parameters
    ----------
    draw
        The Hypothesis draw function.
    st_
        A callable that accepts a ``max_size`` keyword argument and returns
        a strategy.
    max_size
        Total element budget shared across all draws.
    '''

    max_size = draw(st.integers(min_value=0, max_value=max_size))

    def _draw_content() -> _T | None:
        nonlocal max_size
        if max_size == 0:
            return None
        result = draw(st_(max_size=max_size))
        max_size -= len(result)
        return result

    return _draw_content
