import numpy as np
from hypothesis import strategies as st

from awkward.contents import NumpyArray
from hypothesis_awkward.strategies.numpy import numpy_arrays


def numpy_array_contents(
    *,
    dtypes: st.SearchStrategy[np.dtype] | None = None,
    allow_nan: bool = True,
    min_size: int = 0,
    max_size: int = 10,
    unique: bool = False,
) -> st.SearchStrategy[NumpyArray]:
    """Strategy for [`ak.contents.NumpyArray`][] instances.

    Parameters
    ----------
    dtypes
        A strategy for NumPy scalar dtypes. If `None`, the default strategy that
        generates any scalar dtype supported by Awkward Array is used.
    allow_nan
        No `NaN`/`NaT` values are generated if `False`.
    min_size
        Minimum number of elements.
    max_size
        Maximum number of elements.
    unique
        No two elements are equal if `True`. The achievable size is bounded by the
        number of distinct values the dtype can represent; when `dtypes` is `None`,
        dtypes that cannot supply `min_size` distinct values (for example `bool` when
        `min_size > 2`) are not drawn. With a given `dtypes`, such a drawn dtype is
        rejected, and Hypothesis raises `Unsatisfiable` if every drawn dtype is
        rejected.

    Returns
    -------
    NumpyArray

    Examples
    --------
    >>> c = numpy_array_contents().example()
    >>> isinstance(c, NumpyArray)
    True
    """
    return numpy_arrays(
        dtype=dtypes,
        allow_structured=False,
        allow_nan=allow_nan,
        max_dims=1,
        min_size=min_size,
        max_size=max_size,
        unique=unique,
    ).map(NumpyArray)
