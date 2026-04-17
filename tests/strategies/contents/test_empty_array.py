from hypothesis import Phase, find, given, settings
from hypothesis import strategies as st

import awkward as ak
from hypothesis_awkward import strategies as st_ak
from hypothesis_awkward.util import iter_leaf_contents


@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """Assert the results of `empty_array_contents()`."""
    result = data.draw(st_ak.contents.empty_array_contents(), label='result')
    assert isinstance(result, ak.contents.EmptyArray)
    assert len(result) == 0


def test_draw_from_contents() -> None:
    """Assert that EmptyArray can be drawn from `contents()`."""
    find(
        st_ak.contents.contents(),
        lambda c: any(
            isinstance(leaf, ak.contents.EmptyArray) for leaf in iter_leaf_contents(c)
        ),
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )
