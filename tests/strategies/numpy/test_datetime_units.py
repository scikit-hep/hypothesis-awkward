import re
from typing import TypedDict, cast

import numpy as np
import pytest
from hypothesis import find, given
from hypothesis import strategies as st

import awkward as ak
from hypothesis_awkward.strategies.numpy.datetime_unit import datetime_units
from tests.find_settings import FIND
from tests.funcs import assert_kwargs_match_signature

ALL_BASE_UNITS = ('Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns', 'ps', 'fs', 'as')


class DatetimeUnitsKwargs(TypedDict, total=False):
    """Options for `datetime_units()` strategy."""

    allow_multiples: bool


DEFAULTS = DatetimeUnitsKwargs(allow_multiples=True)


def test_kwargs_match_signature() -> None:
    """Assert the option declarations agree with `datetime_units()`."""
    assert_kwargs_match_signature(
        func=datetime_units,
        kwargs_cls=DatetimeUnitsKwargs,
        defaults=DEFAULTS,
    )


@st.composite
def datetime_units_kwargs(draw: st.DrawFn) -> DatetimeUnitsKwargs:
    """Strategy for options for `datetime_units()` strategy."""
    optional_independent = draw(
        st.fixed_dictionaries({}, optional={'allow_multiples': st.booleans()})
    )
    return cast(DatetimeUnitsKwargs, optional_independent)


@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """Assert the results of `datetime_units()`."""
    # Draw options
    kwargs = data.draw(datetime_units_kwargs(), label='kwargs')

    # Call the test subject
    result = data.draw(datetime_units(**kwargs), label='result')

    allow_multiples = kwargs.get('allow_multiples', DEFAULTS['allow_multiples'])

    # Assert the unit has no multiplier unless multiples are allowed.
    if not allow_multiples:
        assert re.fullmatch(r'[A-Za-z]+', result)

    # Assert that NumPy and Awkward Array accept the unit for both kinds.
    for kind in ('datetime64', 'timedelta64'):
        dtype = np.dtype(f'{kind}[{result}]')
        ak.from_numpy(np.array([], dtype=dtype))

        # Assert the multiplier is 1, or at least 1 if multiples are allowed. NumPy
        # also accepts 0.
        _, count = np.datetime_data(dtype)
        if allow_multiples:
            assert count >= 1
        else:
            assert count == 1


@pytest.mark.parametrize('unit', ALL_BASE_UNITS)
def test_draw_unit(unit: str) -> None:
    """Assert that each unit can be drawn."""
    find(datetime_units(), lambda u: u == unit, settings=FIND)


def test_draw_multiples() -> None:
    """Assert that multiples of units, e.g., `'10m'`, can be drawn."""
    find(datetime_units(), lambda u: _multiplier(u) > 1, settings=FIND)


# TODO: Unskip when the cap in `datetime_units()` is removed.
# https://github.com/scikit-hep/awkward/issues/4581
@pytest.mark.skip(reason='multipliers are capped at 999 until awkward#4581 is fixed')
def test_draw_max_multiplier() -> None:  # pragma: no cover
    """Assert that the largest multiplier NumPy accepts can be drawn."""
    find(datetime_units(), lambda u: _multiplier(u) == 2**31 - 1, settings=FIND)


def test_draw_multiplier_one() -> None:
    """Assert that units with the multiplier 1 written, e.g., `'1m'`, can be drawn."""
    find(
        datetime_units(),
        lambda u: bool(re.fullmatch(r'1[A-Za-z]+', u)),
        settings=FIND,
    )


def test_shrink_to_us() -> None:
    """Assert that `datetime_units()` shrinks to `'us'`."""
    result = find(datetime_units(), lambda _: True, settings=FIND)
    assert result == 'us'


def _multiplier(unit: str) -> int:
    """The multiplier of `unit` as NumPy reads it, e.g., 10 for `'10m'`."""
    _, count = np.datetime_data(np.dtype(f'datetime64[{unit}]'))
    return count
