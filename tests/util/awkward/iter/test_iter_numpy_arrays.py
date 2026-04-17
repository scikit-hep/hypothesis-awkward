import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from awkward.contents import (
    ListArray,
    ListOffsetArray,
    NumpyArray,
    RegularArray,
)
from hypothesis_awkward import strategies as st_ak
from hypothesis_awkward.util import iter_contents, iter_numpy_arrays


@given(data=st.data())
def test_iter_numpy_arrays(data: st.DataObject) -> None:
    """Verify iter_numpy_arrays yields NumpyArray leaf data."""
    a = data.draw(st_ak.constructors.arrays(allow_virtual=False), label='array')
    exclude_string = data.draw(st.booleans(), label='exclude_string')
    exclude_bytestring = data.draw(st.booleans(), label='exclude_bytestring')

    result = list(
        iter_numpy_arrays(
            a, exclude_string=exclude_string, exclude_bytestring=exclude_bytestring
        )
    )

    assert all(isinstance(arr, np.ndarray) for arr in result)

    # Top-level NumpyArray data is in result
    result_ids = {id(arr) for arr in result}
    if isinstance(a.layout, NumpyArray):
        assert id(a.layout.data) in result_ids

    # NumpyArray children of list-type arrays are in result
    for c in iter_contents(
        a, string_as_leaf=exclude_string, bytestring_as_leaf=exclude_bytestring
    ):
        if not isinstance(c, (ListOffsetArray, ListArray, RegularArray)):
            continue
        if exclude_string and c.parameter('__array__') == 'string':
            continue
        if exclude_bytestring and c.parameter('__array__') == 'bytestring':
            continue
        if isinstance(c.content, NumpyArray):
            assert id(c.content.data) in result_ids
