import numpy as np

import awkward as ak
from awkward.contents import Content

from .iter import iter_numpy_arrays


def any_nan_nat_in_awkward_array(a: ak.Array | Content, /) -> bool:
    """`True` if Awkward Array contains any `NaN` or `NaT` values, else `False`.

    Parameters
    ----------
    a
        An Awkward Array.

    Returns
    -------
    bool
        `True` if `a` contains any `NaN` or `NaT` values, else `False`.

    Examples
    --------
    >>> a = ak.Array([1.0, 2.0, np.nan])
    >>> any_nan_nat_in_awkward_array(a)
    True

    >>> a = ak.Array([1.0, 2.0, 3.0])
    >>> any_nan_nat_in_awkward_array(a)
    False

    >>> a = ak.Array([{'x': 1.0, 'y': np.nan}, {'x': 2.0, 'y': 3.0}])
    >>> any_nan_nat_in_awkward_array(a)
    True
    """
    return any_nan_in_awkward_array(a) or any_nat_in_awkward_array(a)


def any_nan_in_awkward_array(a: ak.Array | Content, /) -> bool:
    """`True` if Awkward Array contains any `NaN` values, else `False`.

    Parameters
    ----------
    a
        An Awkward Array.

    Returns
    -------
    bool
        `True` if `a` contains any `NaN` values, else `False`.

    Examples
    --------
    >>> a = ak.Array([1.0, 2.0, np.nan])
    >>> any_nan_in_awkward_array(a)
    True

    >>> a = ak.Array([1.0, 2.0, 3.0])
    >>> any_nan_in_awkward_array(a)
    False

    >>> a = ak.Array([{'x': 1.0, 'y': np.nan}, {'x': 2.0, 'y': 3.0}])
    >>> any_nan_in_awkward_array(a)
    True
    """
    for arr in iter_numpy_arrays(a):
        if arr.dtype.kind in {'f', 'c'} and np.any(np.isnan(arr)):
            return True
    return False


def any_nat_in_awkward_array(a: ak.Array | Content, /) -> bool:
    """`True` if Awkward Array contains any `NaT` values, else `False`.

    Parameters
    ----------
    a
        An Awkward Array.

    Returns
    -------
    bool
        `True` if `a` contains any `NaT` values, else `False`.

    Examples
    --------
    >>> a = ak.Array(np.array(['2020-01-01', 'NaT'], dtype='datetime64[D]'))
    >>> any_nat_in_awkward_array(a)
    True

    >>> a = ak.Array(np.array(['2020-01-01', '2020-01-02'], dtype='datetime64[D]'))
    >>> any_nat_in_awkward_array(a)
    False
    """
    for arr in iter_numpy_arrays(a):
        if arr.dtype.kind in {'m', 'M'} and np.any(np.isnat(arr)):
            return True
    return False
