from typing import Any, TypedDict, cast

import numpy as np
import pytest
from hypothesis import find, given
from hypothesis import strategies as st

import hypothesis_awkward.strategies as st_ak
from hypothesis_awkward._util import any_nan_nat_in_numpy_array
from hypothesis_awkward.strategies.numpy.element import numpy_array_elements
from tests.find_settings import FIND
from tests.funcs import assert_kwargs_match_signature


class NumpyArrayElementsKwargs(TypedDict, total=False):
    """Options for `numpy_array_elements()` strategy."""

    allow_nan: bool


DEFAULTS = NumpyArrayElementsKwargs(allow_nan=True)


def test_kwargs_match_signature() -> None:
    """Assert the option declarations agree with `numpy_array_elements()`."""
    assert_kwargs_match_signature(
        func=numpy_array_elements,
        exclude={'dtype'},
        kwargs_cls=NumpyArrayElementsKwargs,
        defaults=DEFAULTS,
    )


@st.composite
def numpy_array_elements_kwargs(draw: st.DrawFn) -> NumpyArrayElementsKwargs:
    """Strategy for options for `numpy_array_elements()` strategy."""
    optional_independent = draw(
        st.fixed_dictionaries({}, optional={'allow_nan': st.booleans()})
    )
    return cast(NumpyArrayElementsKwargs, optional_independent)


@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """Assert the results of `numpy_array_elements()`."""
    # Draw the arguments
    allow_array = data.draw(st.booleans(), label='allow_array')
    dtype = data.draw(st_ak.numpy.numpy_dtypes(allow_array=allow_array), label='dtype')
    kwargs = data.draw(numpy_array_elements_kwargs(), label='kwargs')

    # Call the test subject
    result = data.draw(numpy_array_elements(dtype, **kwargs), label='result')

    allow_nan = kwargs.get('allow_nan', DEFAULTS['allow_nan'])

    # Assert the result is an element of an array of `dtype`.
    n = np.array(result, dtype=dtype)
    assert n.dtype == dtype
    assert n.shape == ()

    if not allow_nan:
        assert not any_nan_nat_in_numpy_array(n)

    for x in _scalars(n):
        # Assert every scalar can be hashed, e.g., to be put in a set.
        hash(x)


@pytest.mark.parametrize(
    'dtype',
    [
        'float64',
        'complex128',
        'datetime64[m]',
        'datetime64[10m]',
        'timedelta64[m]',
        'timedelta64[10m]',
    ],
)
def test_draw_nan(dtype: str) -> None:
    """Assert that `NaN` and `NaT` can be drawn by default."""
    find(
        numpy_array_elements(np.dtype(dtype)),
        lambda v: any_nan_nat_in_numpy_array(np.array(v, dtype=dtype)),
        settings=FIND,
    )


def _scalars(n: np.ndarray) -> list[Any]:
    """Return the scalars of `n`, field by field for a structured array.

    A structured element (`np.void`) cannot be hashed, but its fields can.
    """
    if n.dtype.names is None:
        return list(n.ravel())
    return [x for name in n.dtype.names for x in _scalars(n[name])]
