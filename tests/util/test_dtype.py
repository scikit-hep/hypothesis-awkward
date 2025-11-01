import numpy as np
from hypothesis import given, note

import hypothesis_awkward.strategies as st_ak
from hypothesis_awkward.util import (
    SUPPORTED_DTYPES,
    n_scalars_in,
    simple_dtype_kinds_in,
    simple_dtypes_in,
)


def _is_dtype_in(s: np.dtype, d: np.dtype) -> bool:
    if (fields := d.fields) is not None:
        return any(_is_dtype_in(s, f[0]) for f in fields.values())
    if (subdtype := d.subdtype) is not None:
        return _is_dtype_in(s, subdtype[0])
    return s == d


@given(d=st_ak.numpy_dtypes())
def test_simple_dtype_in(d: np.dtype) -> None:
    simple_dtypes = simple_dtypes_in(d)
    for s in simple_dtypes:
        assert s.subdtype is None
        assert s.names is None
        assert _is_dtype_in(s, d)
    for s in set(SUPPORTED_DTYPES) - simple_dtypes:
        assert not _is_dtype_in(s, d)


def _is_kind_in(k: str, d: np.dtype) -> bool:
    if (fields := d.fields) is not None:
        return any(_is_kind_in(k, f[0]) for f in fields.values())
    if (subdtype := d.subdtype) is not None:
        return _is_kind_in(k, subdtype[0])
    return d.kind == k


# https://numpy.org/doc/2.0/reference/generated/numpy.dtype.kind.html
ALL_SIMPLE_DTYPE_KINDS = set('biufcmMOSU')


@given(d=st_ak.numpy_dtypes())
def test_simple_dtype_kinds_in(d: np.dtype) -> None:
    kinds = simple_dtype_kinds_in(d)
    assert 'V' not in kinds
    for k in kinds:
        assert _is_kind_in(k, d)
    for k in ALL_SIMPLE_DTYPE_KINDS - kinds:
        assert not _is_kind_in(k, d)


@given(dtype=st_ak.numpy_dtypes())
def test_n_scalars_in(dtype: np.dtype) -> None:
    num = n_scalars_in(dtype)
    note(f'{num=}')

    zeros = np.zeros((), dtype=dtype)
    note(f'{zeros=}')

    def _flatten(x: np.ndarray) -> list:
        kind = x.dtype.kind
        match kind:
            case 'V':
                return [i for n in x.dtype.names for i in _flatten(x[n])]
            case _:
                return x.flatten().tolist()

    flat = _flatten(zeros)
    note(f'{flat=}')

    assert len(flat) == num
