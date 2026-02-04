from typing import Any, Generic, Mapping, Protocol, TypeVar

from hypothesis import strategies as st

T = TypeVar('T')
K = TypeVar('K', bound=Mapping[str, Any])


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


class Opts(Generic[K]):
    '''Drawn options with resettable recorders.

    Wraps a kwargs dict that may contain ``RecordDraws`` values.
    Call ``reset()`` before each draw of the strategy under test to
    clear stale recorded values from previous Hypothesis attempts.
    Within a single ``@given`` run, Hypothesis reuses the same
    ``RecordDraws`` instances across attempts, so without ``reset()``
    the ``drawn`` lists would accumulate values from earlier attempts.

    Examples
    --------
    >>> recorder = RecordDraws(st.integers())
    >>> opts = Opts({'values': recorder})
    >>> _ = recorder.example()
    >>> len(recorder.drawn) > 0
    True
    >>> opts.reset()
    >>> recorder.drawn
    []
    '''

    def __init__(self, kwargs: K) -> None:
        self._kwargs = kwargs

    @property
    def kwargs(self) -> K:
        return self._kwargs

    def reset(self) -> None:
        for v in self._kwargs.values():
            if isinstance(v, RecordDraws):
                v.drawn.clear()
