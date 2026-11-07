from typing import TypedDict, cast

import numpy as np
from hypothesis import find, given
from hypothesis import strategies as st

import awkward as ak
from hypothesis_awkward import strategies as st_ak
from hypothesis_awkward._util import n_scalars_in, simple_dtype_kinds_in
from hypothesis_awkward._util import safe_compare as sc
from tests.find_settings import FIND
from tests.funcs import assert_kwargs_match_signature


@given(name=st_ak.numpy.supported_dtype_names())
def test_supported_dtype_names(name: str) -> None:
    ak.from_numpy(np.array([], dtype=name))


@given(dtype=st_ak.numpy.supported_dtypes())
def test_supported_dtypes(dtype: np.dtype) -> None:
    ak.from_numpy(np.array([], dtype=dtype))


def test_supported_dtypes_shrinks_to_bool() -> None:
    """Assert that supported_dtypes() shrinks to bool."""
    result = find(st_ak.numpy.supported_dtypes(), lambda _: True, settings=FIND)
    assert result == np.dtype('bool')


class NumpyDtypesKwargs(TypedDict, total=False):
    """Options for `numpy_dtypes()` strategy."""

    dtype: np.dtype | st.SearchStrategy[np.dtype] | None
    allow_array: bool
    max_size: int


DEFAULTS = NumpyDtypesKwargs(dtype=None, allow_array=True, max_size=5)


def test_kwargs_match_signature() -> None:
    """Assert the option declarations agree with `numpy_dtypes()`."""
    assert_kwargs_match_signature(
        func=st_ak.numpy.numpy_dtypes, kwargs_cls=NumpyDtypesKwargs, defaults=DEFAULTS
    )


def numpy_dtypes_kwargs() -> st.SearchStrategy[NumpyDtypesKwargs]:
    """Strategy for options for `numpy_dtypes()` strategy."""
    return st.fixed_dictionaries(
        {},
        optional={
            'dtype': st.one_of(
                st.none(),
                st.just(st_ak.numpy.supported_dtypes()),
                st_ak.numpy.supported_dtypes(),
            ),
            'allow_array': st.booleans(),
            'max_size': st.integers(min_value=1, max_value=10),
        },
    ).map(lambda d: cast(NumpyDtypesKwargs, d))


@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """Assert the results of `numpy_dtypes()`."""
    # Draw options
    kwargs = data.draw(numpy_dtypes_kwargs(), label='kwargs')

    # Call the test subject
    result = data.draw(st_ak.numpy.numpy_dtypes(**kwargs), label='dtype')

    # Assert the options were effective
    dtype = kwargs.get('dtype', DEFAULTS['dtype'])
    allow_array = kwargs.get('allow_array', DEFAULTS['allow_array'])
    max_size = kwargs.get('max_size', DEFAULTS['max_size'])

    if dtype is not None and not isinstance(dtype, st.SearchStrategy):
        kinds = simple_dtype_kinds_in(result)
        assert len(kinds) == 1
        assert dtype.kind in kinds
    if not allow_array:
        assert result.names is None  # not structured
    assert n_scalars_in(result) <= sc(max_size)

    # Assert an Awkward Array can be created.
    ak.from_numpy(np.array([], dtype=result))
