"""Strategies for NumPy dtypes and arrays that Awkward Array supports.

`supported_dtypes()` generates scalar dtypes, `numpy_dtypes()` also generates structured
dtypes, and `numpy_arrays()` generates NumPy arrays that can be converted to Awkward
Arrays.
"""

__all__ = [
    'supported_dtypes',
    'numpy_dtypes',
    'numpy_arrays',
]

from .dtype import numpy_dtypes, supported_dtypes
from .numpy import numpy_arrays
