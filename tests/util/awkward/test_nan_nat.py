import math

import numpy as np
from hypothesis import Phase, find, given, settings
from hypothesis import strategies as st

import awkward as ak
from hypothesis_awkward.util import (
    any_nan_in_awkward_array,
    any_nan_nat_in_awkward_array,
    any_nat_in_awkward_array,
    iter_numpy_arrays,
)
from tests.util.awkward.conftest import st_arrays


@given(data=st.data())
def test_any_nan_nat_in_awkward_array(data: st.DataObject) -> None:
    '''Verify result matches element-by-element iteration.'''
    allow_nan = data.draw(st.booleans())
    a = data.draw(st_arrays(allow_nan=allow_nan))
    actual = any_nan_nat_in_awkward_array(a)
    if allow_nan:
        expected = _has_nan_nat_via_iteration(a)
    else:
        expected = False
    assert actual == expected


@given(data=st.data())
def test_any_nan_in_awkward_array(data: st.DataObject) -> None:
    '''Verify result matches element-by-element iteration.'''
    allow_nan = data.draw(st.booleans())
    a = data.draw(st_arrays(allow_nan=allow_nan))
    actual = any_nan_in_awkward_array(a)
    if allow_nan:
        expected = _has_nan_via_iteration(a)
    else:
        expected = False
    assert actual == expected


@given(data=st.data())
def test_any_nat_in_awkward_array(data: st.DataObject) -> None:
    '''Verify result matches element-by-element iteration.'''
    allow_nan = data.draw(st.booleans())
    a = data.draw(st_arrays(allow_nan=allow_nan))
    actual = any_nat_in_awkward_array(a)
    if allow_nan:
        expected = _has_nat_via_iteration(a)
    else:
        expected = False
    assert actual == expected


def test_draw_nan() -> None:
    '''Assert that arrays with NaN can be drawn.'''
    find(
        st_arrays(),
        _has_nan_via_iteration,
        settings=settings(max_examples=10000, phases=[Phase.generate]),
    )


def test_draw_nat() -> None:
    '''Assert that arrays with NaT can be drawn.'''
    find(
        st_arrays(),
        _has_nat_via_iteration,
        settings=settings(max_examples=10000, phases=[Phase.generate]),
    )


def _has_nan_nat_via_iteration(a: ak.Array) -> bool:
    '''Check if array contains any NaN or NaT.'''
    return _has_nan_via_iteration(a) or _has_nat_via_iteration(a)


def _has_nan_via_iteration(a: ak.Array) -> bool:
    '''Check for NaN by iterating over underlying numpy arrays.'''
    for arr in iter_numpy_arrays(a):
        match arr.dtype.kind:
            case 'c':
                if any(math.isnan(val.real) or math.isnan(val.imag) for val in arr.flat):
                    return True
            case 'f':
                if any(math.isnan(val) for val in arr.flat):
                    return True
    return False


def _has_nat_via_iteration(a: ak.Array) -> bool:
    '''Check for NaT by iterating over underlying numpy arrays.'''
    for arr in iter_numpy_arrays(a):
        if arr.dtype.kind not in {'m', 'M'}:
            continue
        if any(np.isnat(val) for val in arr.flat):
            return True
    return False
