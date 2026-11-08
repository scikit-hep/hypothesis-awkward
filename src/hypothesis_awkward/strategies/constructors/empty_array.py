from hypothesis import strategies as st

from awkward.contents import EmptyArray


def empty_array_contents() -> st.SearchStrategy[EmptyArray]:
    """Strategy for [`ak.contents.EmptyArray`][] instances.

    Returns
    -------
    EmptyArray

    Examples
    --------
    >>> import awkward as ak
    >>> from hypothesis import given
    >>> import hypothesis_awkward.strategies as st_ak

    An `EmptyArray` has no elements and the type `unknown`:

    >>> @given(c=st_ak.constructors.empty_array_contents())
    ... def test_empty(c):
    ...     assert str(ak.Array(c).type) == '0 * unknown'
    >>> test_empty()
    """
    return st.just(EmptyArray())
