from hypothesis import given
from hypothesis import strategies as st

import awkward as ak
from hypothesis_awkward.util import iter_numpy_arrays
from tests.util.awkward.conftest import st_arrays


@given(data=st.data())
def test_iter_numpy_arrays(data: st.DataObject) -> None:
    '''Verify total element count matches len(ak.flatten()).'''
    a = data.draw(st_arrays())
    total = sum(arr.size for arr in iter_numpy_arrays(a))
    expected = _count_all_leaves(a)
    assert total == expected


def _count_all_leaves(a: ak.Array) -> int:
    '''Count total elements across all leaf fields (handles record arrays).'''
    if a.fields:
        return sum(_count_all_leaves(a[field]) for field in a.fields)
    return len(ak.flatten(a, axis=None))
