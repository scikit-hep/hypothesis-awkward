import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as st_np

from hypothesis_awkward._util import n_scalars_in

from .datetime_unit import datetime_units

# Within each kind, the dtype with a corresponding Python built-in type comes first.
SUPPORTED_INT_DTYPE_NAMES = ('int64', 'int8', 'int16', 'int32')
SUPPORTED_UINT_DTYPE_NAMES = ('uint8', 'uint16', 'uint32', 'uint64')
SUPPORTED_FLOAT_DTYPE_NAMES = ('float64', 'float16', 'float32')
SUPPORTED_COMPLEX_DTYPE_NAMES = ('complex128', 'complex64')


def supported_dtype_names() -> st.SearchStrategy[str]:
    """Strategy for names of NumPy dtypes supported by Awkward Array.

    Datetime and timedelta dtypes are drawn with units, e.g., `datetime64[m]`, and
    multiples of units, e.g., `datetime64[10m]`.

    Examples
    --------
    >>> supported_dtype_names().example()
    '...'
    """
    units = datetime_units()
    return st.one_of(
        st.just('bool'),
        st.sampled_from(SUPPORTED_INT_DTYPE_NAMES),
        st.sampled_from(SUPPORTED_UINT_DTYPE_NAMES),
        st.sampled_from(SUPPORTED_FLOAT_DTYPE_NAMES),
        st.sampled_from(SUPPORTED_COMPLEX_DTYPE_NAMES),
        units.map(lambda u: f'datetime64[{u}]'),
        units.map(lambda u: f'timedelta64[{u}]'),
    )


def supported_dtypes() -> st.SearchStrategy[np.dtype]:
    """Strategy for NumPy dtypes supported by Awkward Array.

    Datetime and timedelta dtypes are drawn with units, e.g., `datetime64[m]`, and
    multiples of units, e.g., `datetime64[10m]`.

    Examples
    --------
    >>> import awkward as ak
    >>> import numpy as np
    >>> from hypothesis import given
    >>> import hypothesis_awkward.strategies as st_ak

    Every drawn dtype can be the dtype of a `NumpyArray`:

    >>> @given(d=st_ak.numpy.supported_dtypes())
    ... def test_numpy_array(d):
    ...     assert ak.contents.NumpyArray(np.zeros(3, dtype=d)).dtype == d
    >>> test_numpy_array()
    """
    return supported_dtype_names().map(np.dtype)


def numpy_dtypes(
    *,
    dtype: np.dtype | st.SearchStrategy[np.dtype] | None = None,
    allow_array: bool = True,
    max_size: int = 5,
) -> st.SearchStrategy[np.dtype]:
    """Strategy for dtypes (simple or array) supported by Awkward Array.

    Examples of simple dtypes are dtype('int32'), dtype('float64')

    Array dtypes include ([('f0', 'i4'), ('f1', 'f8')]). They are dtypes of structured
    NumPy arrays.

    Parameters
    ----------
    dtype
        A simple dtype or a strategy for simple dtypes for determining the type of
        array elements. If `None`, any supported simple dtype is used.
    allow_array
        Generate only simple dtypes if `False`, else array dtypes as well.
    max_size
        Maximum number of scalars in a structured dtype.

    Examples
    --------
    >>> from hypothesis import given
    >>> import hypothesis_awkward.strategies as st_ak

    Exclude structured dtypes:

    >>> @given(d=st_ak.numpy.numpy_dtypes(allow_array=False))
    ... def test_simple(d):
    ...     assert d.names is None
    >>> test_simple()
    """
    if dtype is None:
        dtype = supported_dtypes()
    if not isinstance(dtype, st.SearchStrategy):
        dtype = st.just(dtype)
    if not allow_array:
        return dtype
    return st_np.array_dtypes(
        subtype_strategy=dtype, max_size=max_size, allow_subarrays=True
    ).filter(lambda d: n_scalars_in(d) <= max_size)
