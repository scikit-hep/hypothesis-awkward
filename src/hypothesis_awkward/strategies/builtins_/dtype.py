"""Strategies for NumPy dtypes with corresponding Python built-in types."""

from typing import Any

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as st_np

from hypothesis_awkward.util import BUILTIN_SAFE_DTYPE_NAMES, BUILTIN_SAFE_DTYPES


def builtin_safe_dtype_names() -> st.SearchStrategy[str]:
    """Strategy for names of NumPy dtypes with corresponding Python built-in types.

    Examples
    --------
    >>> builtin_safe_dtype_names().example()
    '...'
    """
    return st.sampled_from(BUILTIN_SAFE_DTYPE_NAMES)


def builtin_safe_dtypes() -> st.SearchStrategy[np.dtype]:
    """Strategy for NumPy dtypes with corresponding Python built-in types.

    Examples
    --------
    >>> builtin_safe_dtypes().example()
    dtype(...)
    """
    return st.sampled_from(BUILTIN_SAFE_DTYPES)


def items_from_dtype(
    dtype: np.dtype, *, allow_nan: bool = True
) -> st.SearchStrategy[Any]:
    """Strategy for Python built-in type values for a given NumPy dtype.

    Parameters
    ----------
    dtype
        The NumPy dtype to generate items for.
    allow_nan
        Generate potentially `NaN` for relevant dtypes if `True`.

    Examples
    --------
    >>> i = items_from_dtype(np.dtype('int64')).example()
    >>> isinstance(i, int)
    True
    """
    return (
        st_np.from_dtype(dtype, allow_nan=allow_nan)
        .map(lambda x: x.item())
        .filter(lambda item: dtype.kind == 'i' or type(item) is not int)
    )
    # Reject if the item is coerced to `int` when `dtype` is not integer.
    # This could happen for `datetime64` and `timedelta64` dtypes.
