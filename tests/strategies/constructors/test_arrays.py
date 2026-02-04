from typing import TypedDict, cast

import numpy as np
from hypothesis import Phase, find, given, note, settings
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak
from hypothesis_awkward.util import (
    any_nan_in_awkward_array,
    any_nan_nat_in_awkward_array,
)

DEFAULT_MAX_LENGTH = 5


class ArraysKwargs(TypedDict, total=False):
    '''Options for `arrays()` strategy.'''

    dtypes: st.SearchStrategy[np.dtype] | None
    allow_nan: bool
    max_length: int


def arrays_kwargs() -> st.SearchStrategy[st_ak.Opts[ArraysKwargs]]:
    '''Strategy for options for `arrays()` strategy.'''
    return (
        st.fixed_dictionaries(
            {},
            optional={
                'dtypes': st.one_of(
                    st.none(),
                    st.just(st_ak.RecordDraws(st_ak.supported_dtypes())),
                ),
                'allow_nan': st.booleans(),
                'max_length': st.integers(min_value=0, max_value=50),
            },
        )
        .map(lambda d: cast(ArraysKwargs, d))
        .map(st_ak.Opts)
    )


@settings(max_examples=200)
@given(data=st.data())
def test_arrays(data: st.DataObject) -> None:
    '''Test that `arrays()` respects all its options.'''
    # Draw options
    opts = data.draw(arrays_kwargs(), label='opts')
    opts.reset()

    # Call the test subject
    a = data.draw(st_ak.constructors.arrays(**opts.kwargs), label='a')

    # Assert the result is always an ak.Array backed by NumpyArray
    assert isinstance(a, ak.Array)
    assert isinstance(a.layout, ak.contents.NumpyArray)

    # Assert the layout data is 1-D
    assert len(a.layout.data.shape) == 1

    # Assert the options were effective
    dtypes = opts.kwargs.get('dtypes', None)
    allow_nan = opts.kwargs.get('allow_nan', False)
    max_length = opts.kwargs.get('max_length', DEFAULT_MAX_LENGTH)

    note(f'{a=}')
    note(f'{a.layout.dtype=}')

    assert len(a) <= max_length

    match dtypes:
        case None:
            pass
        case st_ak.RecordDraws():
            drawn_dtype_names = {d.name for d in dtypes.drawn}
            assert a.layout.dtype.name in drawn_dtype_names

    if not allow_nan:
        assert not any_nan_nat_in_awkward_array(a)


def test_draw_empty() -> None:
    '''Assert that empty arrays can be drawn by default.'''
    find(
        st_ak.constructors.arrays(),
        lambda a: len(a) == 0,
        settings=settings(phases=[Phase.generate]),
    )


def test_draw_max_length() -> None:
    '''Assert that arrays with max_length elements can be drawn.'''
    find(
        st_ak.constructors.arrays(),
        lambda a: len(a) == DEFAULT_MAX_LENGTH,
        settings=settings(phases=[Phase.generate]),
    )


def test_draw_nan() -> None:
    '''Assert that arrays with NaN can be drawn when allowed.'''
    float_dtypes = st_ak.supported_dtypes().filter(lambda d: d.kind == 'f')
    find(
        st_ak.constructors.arrays(dtypes=float_dtypes, allow_nan=True),
        any_nan_in_awkward_array,
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_integer_dtype() -> None:
    '''Assert that integer dtype arrays can be drawn.'''
    int_dtypes = st_ak.supported_dtypes().filter(lambda d: d.kind == 'i')
    find(
        st_ak.constructors.arrays(dtypes=int_dtypes),
        lambda a: a.layout.dtype.kind == 'i',
        settings=settings(phases=[Phase.generate]),
    )
