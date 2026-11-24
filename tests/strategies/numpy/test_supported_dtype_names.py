import numpy as np
import pytest
from hypothesis import find, given

import awkward as ak
from awkward.types.numpytype import dtype_to_primitive
from hypothesis_awkward.strategies.numpy.dtype import supported_dtype_names
from tests.find_settings import FIND


@given(name=supported_dtype_names())
def test_properties(name: str) -> None:
    """Assert the results of `supported_dtype_names()`."""
    dtype = np.dtype(name)

    # Assert that Awkward Array's name for the dtype gives the same dtype. The names can
    # differ: NumPy normalizes `datetime64[1m]` to `datetime64[m]`.
    assert np.dtype(dtype_to_primitive(dtype)) == dtype

    ak.from_numpy(np.array([], dtype=dtype))


def test_shrink_to_bool() -> None:
    """Assert that `supported_dtype_names()` shrinks to `'bool'`."""
    result = find(supported_dtype_names(), lambda _: True, settings=FIND)
    assert result == 'bool'


@pytest.mark.parametrize('kind', ['datetime64', 'timedelta64'])
def test_draw_multiples(kind: str) -> None:
    """Assert that multiples of units, e.g., `datetime64[10m]`, can be drawn."""
    find(
        supported_dtype_names(),
        lambda n: n.startswith(kind) and np.datetime_data(np.dtype(n))[1] > 1,
        settings=FIND,
    )
