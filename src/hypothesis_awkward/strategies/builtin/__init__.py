"""Strategies for built-in Python objects in the context of Awkward Array.

These strategies are related to the section of Awkward Array User Guide ["How to convert
to/from Python objects"][ak-user-guide-python].

[ak-user-guide-python]:
https://awkward-array.org/doc/stable/user-guide/how-to-convert-python.html
"""

__all__ = [
    'builtin_safe_dtype_names',
    'builtin_safe_dtypes',
    'items_from_dtype',
    'lists',
    'from_list',
]

from .dtype import builtin_safe_dtype_names, builtin_safe_dtypes, items_from_dtype
from .list_ import from_list, lists
