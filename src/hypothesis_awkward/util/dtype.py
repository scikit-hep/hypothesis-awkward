"""NumPy dtype utilities for Awkward Array.

Attributes
----------
BUILTIN_SAFE_DTYPE_NAMES
    Names of NumPy dtypes with corresponding Python built-in types. Sorted
    for optimal shrinking from simple to complex dtypes.
    Note that ``datetime64[us]`` isn't entirely safe. For example, a value
    with the year zero is coerced to ``int``.
BUILTIN_SAFE_DTYPES
    NumPy dtypes with corresponding Python built-in types. Sorted for optimal shrinking
    from simple to complex dtypes.
SUPPORTED_DTYPE_NAMES
    Names of all NumPy scalar dtypes supported by Awkward Array.
SUPPORTED_DTYPES
    All NumPy scalar dtypes supported by Awkward Array.
"""

from collections.abc import Mapping

import numpy as np

from awkward.types.numpytype import _primitive_to_dtype_dict, primitive_to_dtype

BUILTIN_SAFE_DTYPE_NAMES = (
    'bool',
    'int64',
    'float64',
    'complex128',
    'datetime64[us]',
    'timedelta64[us]',
)

BUILTIN_SAFE_DTYPES = tuple[np.dtype, ...](
    primitive_to_dtype(name) for name in BUILTIN_SAFE_DTYPE_NAMES
)


def _supported_dtype_names() -> tuple[str, ...]:
    """Return names of NumPy scalar dtypes supported by Awkward Array.

    They are sorted for optimal shrinking from simple to complex dtypes,
    with dtypes corresponding to Python built-in types first, then
    remaining dtypes grouped by kind (int, uint, float, complex,
    datetime, timedelta) and ordered smallest to largest within each.

    Datetime and timedelta units are interleaved and ordered by
    proximity to microseconds (``us``), the built-in-safe unit.

    The set of supported dtypes is derived from Awkward Array's
    internal ``_primitive_to_dtype_dict``; only the ordering is
    controlled here.

    As of this writing::

        (
            'bool',
            'int64',
            'float64',
            'complex128',
            'datetime64[us]',
            'timedelta64[us]',
            'int8',
            'int16',
            'int32',
            'uint8',
            'uint16',
            'uint32',
            'uint64',
            'float16',
            'float32',
            'complex64',
            'datetime64[ms]',
            'timedelta64[ms]',
            'datetime64[ns]',
            'timedelta64[ns]',
            'datetime64[s]',
            'timedelta64[s]',
            'datetime64[ps]',
            'timedelta64[ps]',
            'datetime64[m]',
            'timedelta64[m]',
            'datetime64[fs]',
            'timedelta64[fs]',
            'datetime64[h]',
            'timedelta64[h]',
            'datetime64[as]',
            'timedelta64[as]',
            'datetime64[D]',
            'timedelta64[D]',
            'datetime64[W]',
            'timedelta64[W]',
            'datetime64[M]',
            'timedelta64[M]',
            'datetime64[Y]',
            'timedelta64[Y]',
        )
    """
    # Ordered for optimal shrinking
    DATETIME_UNITS = tuple('us ms ns s ps m fs h as D W M Y'.split())

    # ('bool', 'int8', ...)
    base = tuple(
        n
        for n, d in _primitive_to_dtype_dict.items()
        if d.kind not in ('M', 'm')  # Exclude datetime/timedelta as they need units
    )

    # ('datetime64[us]', 'datetime64[ms]', ...)
    dt = tuple(f'datetime64[{unit}]' for unit in DATETIME_UNITS)

    # ('timedelta64[us]', 'timedelta64[ms]', ...)
    td = tuple(f'timedelta64[{unit}]' for unit in DATETIME_UNITS)

    # Interleave datetime and timedelta: datetime64[us], timedelta64[us], ...
    dt_td = tuple(n for pair in zip(dt, td) for n in pair)

    all_names = base + dt_td
    preferred = set(BUILTIN_SAFE_DTYPE_NAMES)
    rest = sorted(
        (n for n in all_names if n not in preferred),
        key=_dtype_sort_key,
    )
    return BUILTIN_SAFE_DTYPE_NAMES + tuple(rest)


# Sort order for remaining dtypes: int → uint → float → complex → datetime → timedelta,
# smallest to largest within each kind.
_KIND_ORDER = {'b': 0, 'i': 1, 'u': 2, 'f': 3, 'c': 4, 'M': 5, 'm': 5}


def _dtype_sort_key(name: str) -> tuple[int, int]:
    d = primitive_to_dtype(name)
    return (_KIND_ORDER[d.kind], d.itemsize)


# Names of NumPy dtypes supported by Awkward Array
# ('bool', 'int8', 'float16', 'datetime64[ns]', ...)
SUPPORTED_DTYPE_NAMES = _supported_dtype_names()

# NumPy dtypes supported by Awkward Array
# (dtype('bool'), dtype('int8'), dtype('float16'), dtype('datetime64[ns]'), ...)
SUPPORTED_DTYPES = tuple[np.dtype, ...](
    primitive_to_dtype(name) for name in SUPPORTED_DTYPE_NAMES
)


def simple_dtypes_in(d: np.dtype, /) -> set[np.dtype]:
    """Return simple dtypes contained in a (compound) dtype `d`.

    Parameters
    ----------
    d
        A NumPy dtype. It can be a sub-array or structured dtype as well as a simple
        dtype.

    Returns
    -------
    set of np.dtype
        Simple dtypes contained in `d`.

    Examples
    --------
    >>> simple_dtypes_in(np.dtype('int32'))
    {dtype('int32')}

    >>> sorted(simple_dtypes_in(np.dtype([('f0', 'i4'), ('f1', 'f8')])))
    [dtype('int32'), dtype('float64')]
    """
    match d.names, d.kind, d.subdtype, d.fields:
        case None, str(), None, None:
            # Simple dtype
            # e.g., d = dtype('int32')
            return {d}
        case None, 'V', tuple(subdtype), None:
            # Sub-array dtype
            # e.g., d = dtype(('int32', (3, 4)))
            return simple_dtypes_in(subdtype[0])
        case tuple(names), 'V', None, fields if isinstance(fields, Mapping):
            # Structured dtype
            # e.g., d = dtype([('f0', 'i4'), ('f1', 'f8')])
            return {t for n in names for t in simple_dtypes_in(fields[n][0])}
        case _:  # pragma: no cover
            raise TypeError(f'Unexpected dtype: {d}')


def simple_dtype_kinds_in(d: np.dtype, /) -> set[str]:
    """Return character codes of simple dtypes contained in a (compound) dtype `d`.

    Parameters
    ----------
    d
        A NumPy dtype. It can be a sub-array or structured dtype as well as a simple
        dtype.

    Returns
    -------
    set of str
        Character codes of simple dtypes contained in `d`.

    Examples
    --------
    >>> simple_dtype_kinds_in(np.dtype('int32'))
    {'i'}

    >>> sorted(simple_dtype_kinds_in(np.dtype([('f0', 'i4'), ('f1', 'f8')])))
    ['f', 'i']
    """
    return {t.kind for t in simple_dtypes_in(d)}


def n_scalars_in(d: np.dtype, /) -> int:
    """Return the number of scalar values contained in a value of dtype `d`.

    Parameters
    ----------
    d
        A NumPy dtype. It can be a sub-array or structured dtype as well as a simple
        dtype.

    Returns
    -------
    int
        The number of scalar values contained in a value of dtype `d`.

    Examples
    --------
    >>> n_scalars_in(np.dtype('int32'))
    1

    >>> n_scalars_in(np.dtype(('int32', (3, 4))))
    12

    >>> n_scalars_in(np.dtype([('f0', 'i4'), ('f1', ('f8', (2,)))]))
    3
    """
    match d.names, d.kind, d.subdtype, d.fields:
        case None, str(), None, None:
            # Simple dtype
            # e.g., d = dtype('int32')
            return 1
        case None, 'V', tuple(subdtype), None:
            # Sub-array dtype
            # e.g., d = dtype(('int32', (3, 4)))
            return int(np.prod(subdtype[1])) * n_scalars_in(subdtype[0])
        case tuple(names), 'V', None, fields if isinstance(fields, Mapping):
            # Structured dtype
            # e.g., d = dtype([('f0', 'i4'), ('f1', ('f8', (2,)))])
            return sum(n_scalars_in(fields[n][0]) for n in names)
        case _:  # pragma: no cover
            raise TypeError(f'Unexpected dtype: {d}')
