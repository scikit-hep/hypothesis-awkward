from hypothesis import strategies as st

from awkward.contents import EmptyArray


def empty_array_contents() -> st.SearchStrategy[EmptyArray]:
    '''Strategy for EmptyArray content.'''
    return st.just(EmptyArray())
