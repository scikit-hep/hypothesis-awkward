from typing import Protocol, TypeVar

from hypothesis import strategies as st

T = TypeVar('T')


class _DrawableData(Protocol):
    '''Protocol for the ``data`` argument of ``do_draw``.'''

    def draw(self, strategy: st.SearchStrategy[T]) -> T: ...


class RecordDraws(st.SearchStrategy[T]):
    '''Wrap a strategy to store all drawn values.

    Examples
    --------
    >>> recorder = RecordDraws(st.integers())
    >>> value = recorder.example()
    >>> value in recorder.drawn
    True
    '''

    def __init__(self, base: st.SearchStrategy[T]) -> None:
        super().__init__()
        self.drawn: list[T] = []
        self._base = base

    def do_draw(self, data: _DrawableData) -> T:
        value = data.draw(self._base)
        self.drawn.append(value)
        return value
