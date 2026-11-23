import re

import numpy as np
import pytest
from hypothesis import find, given
from hypothesis import strategies as st

import awkward as ak
from hypothesis_awkward.strategies.numpy.datetime_unit import datetime_units
from tests.find_settings import FIND

ALL_BASE_UNITS = ('Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns', 'ps', 'fs', 'as')


@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """Assert the results of `datetime_units()`."""
    # Call the test subject
    result = data.draw(datetime_units(), label='result')

    # Assert the unit has no multiplier.
    assert re.fullmatch(r'[A-Za-z]+', result)

    # Assert that NumPy and Awkward Array accept the unit for both kinds.
    for kind in ('datetime64', 'timedelta64'):
        dtype = np.dtype(f'{kind}[{result}]')
        ak.from_numpy(np.array([], dtype=dtype))

        # Assert the multiplier is 1.
        _, count = np.datetime_data(dtype)
        assert count == 1


@pytest.mark.parametrize('unit', ALL_BASE_UNITS)
def test_draw_unit(unit: str) -> None:
    """Assert that each unit can be drawn."""
    find(datetime_units(), lambda u: u == unit, settings=FIND)


def test_shrink_to_us() -> None:
    """Assert that `datetime_units()` shrinks to `'us'`."""
    result = find(datetime_units(), lambda _: True, settings=FIND)
    assert result == 'us'
