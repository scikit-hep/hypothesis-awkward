import numpy as np
from hypothesis import find, given

import awkward as ak
from awkward.types.numpytype import dtype_to_primitive
from hypothesis_awkward.strategies.numpy.dtype import supported_dtype_names
from tests.find_settings import FIND


@given(name=supported_dtype_names())
def test_properties(name: str) -> None:
    """Assert the results of `supported_dtype_names()`."""
    dtype = np.dtype(name)

    # Assert the name is the canonical name Awkward Array gives the dtype.
    assert dtype_to_primitive(dtype) == name

    ak.from_numpy(np.array([], dtype=dtype))


def test_shrink_to_bool() -> None:
    """Assert that `supported_dtype_names()` shrinks to `'bool'`."""
    result = find(supported_dtype_names(), lambda _: True, settings=FIND)
    assert result == 'bool'
