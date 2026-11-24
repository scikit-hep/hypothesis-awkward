import numpy as np
import pytest
from hypothesis import find, given

import awkward as ak
from hypothesis_awkward import strategies as st_ak
from tests.find_settings import FIND


@given(dtype=st_ak.numpy.supported_dtypes())
def test_properties(dtype: np.dtype) -> None:
    """Assert the results of `supported_dtypes()`."""
    ak.from_numpy(np.array([], dtype=dtype))


def test_shrink_to_bool() -> None:
    """Assert that `supported_dtypes()` shrinks to `bool`."""
    result = find(st_ak.numpy.supported_dtypes(), lambda _: True, settings=FIND)
    assert result == np.dtype('bool')


@pytest.mark.parametrize('kind', ['M', 'm'])
def test_draw_multiples(kind: str) -> None:
    """Assert that multiples of units, e.g., `datetime64[10m]`, can be drawn."""
    find(
        st_ak.numpy.supported_dtypes(),
        lambda d: d.kind == kind and np.datetime_data(d)[1] > 1,
        settings=FIND,
    )
