"""Strategies related to NumPy in the context of Awkward Array.

These strategies are related to the section of Awkward Array User Guide ["How to convert
to/from NumPy"][ak-user-guide-numpy].

[ak-user-guide-numpy]:
https://awkward-array.org/doc/stable/user-guide/how-to-convert-numpy.html
"""

__all__ = [
    'supported_dtypes',
    'numpy_dtypes',
    'numpy_arrays',
]

from .dtype import numpy_dtypes, supported_dtypes
from .numpy import numpy_arrays
