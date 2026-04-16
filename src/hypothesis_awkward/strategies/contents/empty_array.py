from hypothesis import strategies as st

from awkward.contents import EmptyArray


def empty_array_contents() -> st.SearchStrategy[EmptyArray]:
    """Strategy for [`ak.contents.EmptyArray`][] instances."""
    return st.just(EmptyArray())
