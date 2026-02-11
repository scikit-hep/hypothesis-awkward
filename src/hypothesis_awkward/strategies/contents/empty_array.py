from hypothesis import strategies as st

import awkward as ak


def empty_array_contents() -> st.SearchStrategy[ak.contents.EmptyArray]:
    '''Strategy for EmptyArray content.'''
    return st.just(ak.contents.EmptyArray())
