from typing import TypedDict, cast

import numpy as np
from hypothesis import find, given
from hypothesis import strategies as st

import awkward as ak
from hypothesis_awkward import strategies as st_ak
from hypothesis_awkward._util import (
    any_nan_in_awkward_array,
    any_nan_nat_in_awkward_array,
    any_nat_in_awkward_array,
    iter_contents,
    iter_numpy_arrays,
)
from tests.find_settings import FIND_NO_SHRINK
from tests.funcs import assert_kwargs_match_signature
from tests.record import RecordDraws


class FromNumpyKwargs(TypedDict, total=False):
    """Options for `from_numpy()` strategy."""

    dtype: np.dtype | st.SearchStrategy[np.dtype] | None
    allow_structured: bool
    allow_nan: bool
    regulararray: bool | None
    max_size: int


DEFAULTS = FromNumpyKwargs(
    dtype=None, allow_structured=True, allow_nan=True, regulararray=None, max_size=10
)


def test_kwargs_match_signature() -> None:
    """Assert the option declarations agree with `from_numpy()`."""
    assert_kwargs_match_signature(
        func=st_ak.from_numpy, kwargs_cls=FromNumpyKwargs, defaults=DEFAULTS
    )


@st.composite
def from_numpy_kwargs(draw: st.DrawFn) -> FromNumpyKwargs:
    """Strategy for options for `from_numpy()` strategy."""
    kwargs = draw(
        st.fixed_dictionaries(
            {},
            optional={
                'dtype': st.one_of(
                    st.none(),
                    st.just(RecordDraws(st_ak.supported_dtypes())),
                    st_ak.supported_dtypes(),
                ),
                'allow_structured': st.booleans(),
                'allow_nan': st.booleans(),
                'regulararray': st.booleans(),
                'max_size': st.integers(min_value=0, max_value=100),
            },
        )
    )

    return cast(FromNumpyKwargs, kwargs)


@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """Assert the results of `from_numpy()`."""
    # Draw options
    kwargs = data.draw(from_numpy_kwargs(), label='kwargs')

    # Call the test subject
    a = data.draw(st_ak.from_numpy(**kwargs), label='a')
    assert isinstance(a, ak.Array)

    # Assert the options were effective
    dtype = kwargs.get('dtype', DEFAULTS['dtype'])
    allow_structured = kwargs.get('allow_structured', DEFAULTS['allow_structured'])
    allow_nan = kwargs.get('allow_nan', DEFAULTS['allow_nan'])
    regulararray = kwargs.get('regulararray', DEFAULTS['regulararray'])
    max_size = kwargs.get('max_size', DEFAULTS['max_size'])

    dtypes = _leaf_dtypes(a)
    structured = _is_structured(a)
    has_nan = any_nan_nat_in_awkward_array(a)
    multi_dimensional = a.ndim > 1
    size = _size(a)

    match dtype:
        case None:
            pass
        case np.dtype():
            assert len(dtypes) == 1
            assert dtype in dtypes
        case RecordDraws():
            drawn_dtypes = {d for d in dtype.drawn}
            assert dtypes <= drawn_dtypes

    if not allow_structured:
        assert not structured

    if not allow_nan:
        assert not has_nan

    if multi_dimensional and regulararray is not None:
        if regulararray:
            assert _has_regular_array(a)
        else:
            assert not _has_regular_array(a)

    assert size <= max_size


def test_draw_structured() -> None:
    """Assert that structured arrays can be drawn by default."""
    find(
        st_ak.from_numpy(),
        lambda a: isinstance(a.layout, ak.contents.RecordArray),
        settings=FIND_NO_SHRINK,
    )


def test_draw_nan() -> None:
    """Assert that arrays with NaN can be drawn when allowed."""
    floating_dtypes = st_ak.supported_dtypes().filter(lambda d: d.kind == 'f')
    find(
        st_ak.from_numpy(dtype=floating_dtypes, allow_nan=True),
        any_nan_in_awkward_array,
        settings=FIND_NO_SHRINK,
    )


def test_draw_nat_datetime64() -> None:
    """Assert that datetime64 arrays with NaT can be drawn when allowed."""
    datetime64_dtypes = st_ak.supported_dtypes().filter(lambda d: d.kind == 'M')
    find(
        st_ak.from_numpy(dtype=datetime64_dtypes, allow_nan=True),
        any_nat_in_awkward_array,
        settings=FIND_NO_SHRINK,
    )


def test_draw_nat_timedelta64() -> None:
    """Assert that timedelta64 arrays with NaT can be drawn when allowed."""
    timedelta64_dtypes = st_ak.supported_dtypes().filter(lambda d: d.kind == 'm')
    find(
        st_ak.from_numpy(dtype=timedelta64_dtypes, allow_nan=True),
        any_nat_in_awkward_array,
        settings=FIND_NO_SHRINK,
    )


def test_draw_empty() -> None:
    """Assert that empty arrays can be drawn by default."""
    find(st_ak.from_numpy(), lambda a: len(a) == 0, settings=FIND_NO_SHRINK)


def test_draw_regulararray() -> None:
    """Assert that RegularArray layout can be drawn with regulararray=True."""
    find(
        st_ak.from_numpy(allow_structured=False, regulararray=True),
        lambda a: isinstance(a.layout, ak.contents.RegularArray),
        settings=FIND_NO_SHRINK,
    )


def test_draw_max_size() -> None:
    """Assert that arrays with max_size elements can be drawn by default."""
    find(
        st_ak.from_numpy(allow_structured=False),
        lambda a: _size(a) == DEFAULTS['max_size'],
        settings=FIND_NO_SHRINK,
    )


def _leaf_dtypes(a: ak.Array) -> set[np.dtype]:
    """Dtypes of leaf NumPy arrays contained in `a`."""
    return {arr.dtype for arr in iter_numpy_arrays(a)}


def _is_structured(a: ak.Array) -> bool:
    """Check if `a` is a structured array."""
    layout = a.layout
    if isinstance(layout, (ak.contents.NumpyArray, ak.contents.RegularArray)):
        return False
    assert isinstance(layout, ak.contents.RecordArray)  # structured array
    return True


def _has_regular_array(a: ak.Array) -> bool:
    """Check if any content in the layout tree is a RegularArray."""
    return any(isinstance(n, ak.contents.RegularArray) for n in iter_contents(a))


def _size(a: ak.Array) -> int:
    """Total size of all leaf NumPy arrays contained in `a`."""
    return sum(arr.size for arr in iter_numpy_arrays(a))
