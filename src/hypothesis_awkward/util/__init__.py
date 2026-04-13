"""Utility functions for property-based test assertions, etc."""

__all__ = [
    'any_nan_nat_in_awkward_array',
    'any_nan_in_awkward_array',
    'any_nat_in_awkward_array',
    'get_contents',
    'is_string_or_bytestring_leaf',
    'is_string_leaf',
    'is_bytestring_leaf',
    'iter_contents',
    'iter_leaf_contents',
    'iter_numpy_arrays',
    'leaf_size',
    'content_size',
    'any_nan_nat_in_numpy_array',
    'any_nan_in_numpy_array',
    'any_nat_in_numpy_array',
    'simple_dtypes_in',
    'simple_dtype_kinds_in',
    'n_scalars_in',
    'safe_compare',
    'safe_max',
    'safe_min',
    'CountdownDrawer',
    'BUILTIN_SAFE_DTYPE_NAMES',
    'BUILTIN_SAFE_DTYPES',
    'SUPPORTED_DTYPES',
    'SUPPORTED_DTYPE_NAMES',
    'LeafContent',
    '_StWithMinMaxSize',
]

from .awkward import (
    LeafContent,
    any_nan_in_awkward_array,
    any_nan_nat_in_awkward_array,
    any_nat_in_awkward_array,
    content_size,
    get_contents,
    is_bytestring_leaf,
    is_string_leaf,
    is_string_or_bytestring_leaf,
    iter_contents,
    iter_leaf_contents,
    iter_numpy_arrays,
    leaf_size,
)
from .draw import CountdownDrawer, _StWithMinMaxSize
from .dtype import (
    BUILTIN_SAFE_DTYPE_NAMES,
    BUILTIN_SAFE_DTYPES,
    SUPPORTED_DTYPE_NAMES,
    SUPPORTED_DTYPES,
    n_scalars_in,
    simple_dtype_kinds_in,
    simple_dtypes_in,
)
from .numpy import (
    any_nan_in_numpy_array,
    any_nan_nat_in_numpy_array,
    any_nat_in_numpy_array,
)
from .safe import safe_compare, safe_max, safe_min
