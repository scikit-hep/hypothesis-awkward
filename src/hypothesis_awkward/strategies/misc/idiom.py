from typing import Optional, TypeVar

from hypothesis import strategies as st

_T = TypeVar('_T')


def none_or(st_: st.SearchStrategy[_T]) -> st.SearchStrategy[Optional[_T]]:
    """A strategy for `None` or values from another strategy.

    >>> v = none_or(st.integers()).example()
    >>> v is None or isinstance(v, int)
    True
    """
    return st.one_of(st.none(), st_)
