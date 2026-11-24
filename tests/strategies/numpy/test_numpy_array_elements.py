from typing import Any, TypedDict

import numpy as np
import pytest
from hypothesis import find, given
from hypothesis import strategies as st

import hypothesis_awkward.strategies as st_ak
from hypothesis_awkward._util import any_nan_nat_in_numpy_array
from hypothesis_awkward.strategies.numpy.datetime_unit import datetime_units
from hypothesis_awkward.strategies.numpy.element import numpy_array_elements
from tests.find_settings import FIND
from tests.funcs import assert_kwargs_match_signature

INT64_MAX = np.iinfo(np.int64).max


class NumpyArrayElementsKwargs(TypedDict, total=False):
    """Options for `numpy_array_elements()` strategy."""


DEFAULTS = NumpyArrayElementsKwargs()


def test_kwargs_match_signature() -> None:
    """Assert the option declarations agree with `numpy_array_elements()`."""
    assert_kwargs_match_signature(
        func=numpy_array_elements,
        exclude={'dtype', 'allow_nan'},
        kwargs_cls=NumpyArrayElementsKwargs,
        defaults=DEFAULTS,
    )


@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """Assert the results of `numpy_array_elements()`."""
    # Draw the arguments
    allow_array = data.draw(st.booleans(), label='allow_array')
    dtype = data.draw(st_ak.numpy.numpy_dtypes(allow_array=allow_array), label='dtype')
    allow_nan = data.draw(st.booleans(), label='allow_nan')

    # Call the test subject
    result = data.draw(numpy_array_elements(dtype, allow_nan=allow_nan), label='result')

    # Assert the result is an element of an array of `dtype`.
    n = np.array(result, dtype=dtype)
    assert n.dtype == dtype
    assert n.shape == ()

    if not allow_nan:
        assert not any_nan_nat_in_numpy_array(n)

    for x in _scalars(n):
        # Assert every scalar can be hashed, e.g., to be put in a set.
        hash(x)

        _assert_datetime_fits(x)


@given(data=st.data())
def test_datetime_multiples_fit(data: st.DataObject) -> None:
    """Assert `datetime64` values with multiples of units fit, in fields too.

    `test_properties` draws these dtypes too rarely, e.g., as fields of structured
    dtypes, to cover them.
    """
    allow_array = data.draw(st.booleans(), label='allow_array')
    dtype = data.draw(
        st_ak.numpy.numpy_dtypes(dtype=_datetime_multiples(), allow_array=allow_array),
        label='dtype',
    )
    allow_nan = data.draw(st.booleans(), label='allow_nan')

    result = data.draw(numpy_array_elements(dtype, allow_nan=allow_nan), label='result')

    for x in _scalars(np.array(result, dtype=dtype)):
        _assert_datetime_fits(x)


@pytest.mark.parametrize('dtype', ['float64', 'datetime64[m]', 'datetime64[10m]'])
def test_draw_nan(dtype: str) -> None:
    """Assert that `NaN` and `NaT` can be drawn if `allow_nan` is `True`."""
    find(
        numpy_array_elements(np.dtype(dtype), allow_nan=True),
        lambda v: any_nan_nat_in_numpy_array(np.array(v, dtype=dtype)),
        settings=FIND,
    )


@pytest.mark.parametrize('sign', [1, -1])
def test_draw_datetime_multiple_bound(sign: int) -> None:
    """Assert that the largest values of `datetime64[10m]` that fit can be drawn."""
    limit = INT64_MAX // 10
    find(
        numpy_array_elements(np.dtype('datetime64[10m]'), allow_nan=False),
        lambda v: int(np.array(v).view(np.int64)) == sign * limit,
        settings=FIND,
    )


@pytest.mark.parametrize(
    ('allow_nan', 'expected'), [(True, 'NaT'), (False, '1970-01-01T00:00')]
)
def test_shrink_datetime_multiple(allow_nan: bool, expected: str) -> None:
    """Assert that `datetime64[10m]` values shrink to `NaT`, else to the epoch.

    The same as `st_np.from_dtype()` for units, e.g., `datetime64[m]`.
    """
    dtype = np.dtype('datetime64[10m]')
    result = find(
        numpy_array_elements(dtype, allow_nan=allow_nan), lambda _: True, settings=FIND
    )
    assert str(result) == expected
    assert result.dtype == dtype


def _datetime_multiples() -> st.SearchStrategy[np.dtype]:
    """Strategy for `datetime64` dtypes with multiples of units, e.g., `[10m]`."""
    dtypes = datetime_units().map(lambda u: np.dtype(f'datetime64[{u}]'))
    return dtypes.filter(lambda d: np.datetime_data(d)[1] > 1)


def _assert_datetime_fits(x: Any) -> None:
    """Assert that a `datetime64` value times its unit's multiple fits in `int64`."""
    if isinstance(x, np.datetime64) and not np.isnat(x):
        _, num = np.datetime_data(x.dtype)
        assert abs(int(np.array(x).view(np.int64))) * num <= INT64_MAX


def _scalars(n: np.ndarray) -> list[Any]:
    """Return the scalars of `n`, field by field for a structured array.

    A structured element (`np.void`) cannot be hashed, but its fields can.
    """
    if n.dtype.names is None:
        return list(n.ravel())
    return [x for name in n.dtype.names for x in _scalars(n[name])]
