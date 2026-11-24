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
    # Collect `dtype` and the dtypes nested in it, each before those nested in it. The
    # dtypes are identified by position, not by value: NumPy considers some different
    # dtypes equal, e.g., `datetime64[1000ms]` and `datetime64[s]`.
    dtypes = [dtype]
    nested = [list[int]()]  # positions in `dtypes` of the dtypes nested in each
    stack = [0]
    while stack:
        i = stack.pop()
        for d in _nested_dtypes(dtypes[i]):
            nested[i].append(len(dtypes))
            stack.append(len(dtypes))
            dtypes.append(d)
            nested.append([])

    # Build the strategies in reverse, so that the strategies for the dtypes nested in
    # a dtype are built before its own.
    strategies = dict[int, st.SearchStrategy[Any]]()
    for i in reversed(range(len(dtypes))):
        d = dtypes[i]
        if d.fields is not None:  # structured: one value per field
            strategies[i] = st.tuples(*(strategies[j] for j in nested[i]))
        elif d.subdtype is not None:  # subarray field, e.g., ('<M8[10m]', (2,))
            sub, shape = d.subdtype
            strategies[i] = st_np.arrays(sub, shape, elements=strategies[nested[i][0]])
        else:
            strategies[i] = _st_simple(d, allow_nan=allow_nan)
    return strategies[0]


def _nested_dtypes(dtype: np.dtype) -> list[np.dtype]:
    """The dtypes of the fields of `dtype`, or the base dtype of a subarray `dtype`."""
    if dtype.fields is not None:
        fields = dtype.fields
        names = dtype.names or ()
        return [fields[n][0] for n in names]
    if dtype.subdtype is not None:
        sub, _ = dtype.subdtype
        return [sub]
    return []


def _st_simple(dtype: np.dtype, *, allow_nan: bool) -> st.SearchStrategy[Any]:
    """Strategy for values of a simple `dtype`."""
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
