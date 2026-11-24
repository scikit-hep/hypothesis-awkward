from typing import Any

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as st_np


def numpy_array_elements(
    dtype: np.dtype, *, allow_nan: bool = True
) -> st.SearchStrategy[Any]:
    """Strategy for elements of NumPy arrays of `dtype`, one field at a time.

    Like `st_np.from_dtype()`, but every drawn scalar is hashable.

    Parameters
    ----------
    dtype
        The dtype of the array, simple or structured.
    allow_nan
        Draw `NaN` and `NaT` for relevant dtypes if `True`.
    """
    if dtype.fields is not None:  # structured: one value per field
        fields = dtype.fields
        names = dtype.names or ()
        return st.tuples(
            *(numpy_array_elements(fields[n][0], allow_nan=allow_nan) for n in names)
        )
    if dtype.subdtype is not None:  # subarray field, e.g., ('<M8[10m]', (2,))
        sub, shape = dtype.subdtype
        return st_np.arrays(
            sub, shape, elements=numpy_array_elements(sub, allow_nan=allow_nan)
        )
    if dtype.kind == 'M':
        _, num = np.datetime_data(dtype)
        if num > 1:
            # Bound the value so that the value times the multiple, e.g., 10 in
            # `datetime64[10m]`, fits in `int64`. Otherwise, NumPy 2.5 raises
            # `OverflowError` on `hash()` and `repr()`.
            limit = (2**63 - 1) // num
            values: st.SearchStrategy[int | str] = st.integers(-limit, limit)
            if allow_nan:
                values = values | st.just('NaT')
            return values.map(lambda v: np.array(v, dtype=dtype)[()])
    return st_np.from_dtype(dtype, allow_nan=allow_nan)
