from typing import Callable, Generic, ParamSpec, Protocol, TypeVar

from hypothesis import strategies as st

T = TypeVar('T')
P = ParamSpec('P')


class _DrawableData(Protocol):
    """Protocol for the `data` argument of `do_draw`."""

    def draw(self, strategy: st.SearchStrategy[T]) -> T: ...


class RecordDraws(st.SearchStrategy[T]):
    """Wrap a strategy to store all drawn values.

    Examples
    --------
    >>> recorder = RecordDraws(st.integers())
    >>> value = recorder.example()
    >>> value in recorder.drawn
    True
    """

    def __init__(self, base: st.SearchStrategy[T]) -> None:
        super().__init__()
        self.drawn: list[T] = []
        self._base = base

    def do_draw(self, data: _DrawableData) -> T:
        value = data.draw(self._base)
        self.drawn.append(value)
        return value


class RecordCallDraws(Generic[P, T]):
    """Wrap a callable returning a strategy to record all drawn values.

    Each call creates a fresh `RecordDraws` whose draws are aggregated
    into `drawn`.  `reset()` clears all calls and their recorded values.

    Examples
    --------
    >>> recorder = RecordCallDraws(st.just)
    >>> st1 = recorder('a')
    >>> st1.example()
    'a'
    >>> st1.example()
    'a'
    >>> st2 = recorder('b')
    >>> st2.example()
    'b'
    >>> recorder.drawn
    ['a', 'a', 'b']
    >>> recorder.reset()
    >>> recorder.drawn
    []
    """

    def __init__(self, base: Callable[P, st.SearchStrategy[T]]) -> None:
        self._base = base
        self._recorders: list[RecordDraws[T]] = []

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> RecordDraws[T]:
        recorder = RecordDraws(self._base(*args, **kwargs))
        self._recorders.append(recorder)
        return recorder

    @property
    def drawn(self) -> list[T]:
        """All values drawn across all calls, in order."""
        return [v for r in self._recorders for v in r.drawn]

    def reset(self) -> None:
        """Clear all recorded calls and their drawn values."""
        self._recorders.clear()
