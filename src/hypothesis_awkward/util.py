from collections.abc import Mapping

import numpy as np


def simple_dtypes_in(d: np.dtype) -> set[np.dtype]:
    '''Simple dtypes contained in `d`.'''
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
        case _:
            raise TypeError(f'Unexpected dtype: {d}')


def simple_dtype_kinds_in(d: np.dtype) -> set[str]:
    '''Kinds of simple dtypes (e.g. `i`, `f`, `M`) contained in `d`.'''
    return {t.kind for t in simple_dtypes_in(d)}
