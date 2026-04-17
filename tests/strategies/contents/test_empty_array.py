from hypothesis import find, given
from hypothesis import strategies as st

import awkward as ak
from hypothesis_awkward import strategies as st_ak


@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """Assert the results of `empty_array_contents()`."""
    result = data.draw(st_ak.contents.empty_array_contents(), label='result')
    assert isinstance(result, ak.contents.EmptyArray)
    assert len(result) == 0


def test_draw_from_contents() -> None:
    """Assert `contents()` can generate an `EmptyArray` as outermost."""
    find(st_ak.contents.contents(), lambda c: isinstance(c, ak.contents.EmptyArray))
