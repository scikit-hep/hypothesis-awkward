from typing import Optional, TypeVar

from hypothesis import strategies as st

_T = TypeVar('_T')


def none_or(st_: st.SearchStrategy[_T]) -> st.SearchStrategy[Optional[_T]]:
    """A strategy for `None` or values from another strategy.

    Examples
    --------
    >>> from hypothesis import given
    >>> from hypothesis import strategies as st
    >>> import hypothesis_awkward.strategies as st_ak

    Draw an optional argument, here `max_length`, which is `None` or an integer:

    >>> @given(data=st.data())
    ... def test_max_length(data):
    ...     max_length = data.draw(
    ...         st_ak.misc.none_or(st.integers(min_value=0, max_value=5))
    ...     )
    ...     c = data.draw(
    ...         st_ak.constructors.list_offset_array_contents(max_length=max_length)
    ...     )
    ...     assert max_length is None or len(c) <= max_length
    >>> test_max_length()
    """
    return st.one_of(st.none(), st_)
