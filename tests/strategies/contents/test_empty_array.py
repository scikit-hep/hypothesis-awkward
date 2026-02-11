from hypothesis import given
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak


@given(data=st.data())
def test_empty_array_contents(data: st.DataObject) -> None:
    '''Test that `empty_array_contents()` produces an EmptyArray.'''
    result = data.draw(
        st_ak.contents.empty_array_contents(), label='result'
    )
    assert isinstance(result, ak.contents.EmptyArray)
    assert len(result) == 0
