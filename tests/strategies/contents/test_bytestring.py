from typing import Any, TypedDict, cast

import numpy as np
from hypothesis import Phase, find, given, settings
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak
from awkward.contents import ListOffsetArray, NumpyArray
from hypothesis_awkward.util.safe import safe_compare as sc

DEFAULT_MAX_SIZE = 10


class BytestringContentsKwargs(TypedDict, total=False):
    '''Options for `bytestring_contents()` strategy.'''

    min_size: int
    max_size: int


@st.composite
def bytestring_contents_kwargs(
    draw: st.DrawFn,
    chain: st_ak.OptsChain[Any] | None = None,
) -> st_ak.OptsChain[BytestringContentsKwargs]:
    '''Strategy for options for `bytestring_contents()` strategy.'''
    if chain is None:
        chain = st_ak.OptsChain({})

    min_size, max_size = draw(
        st_ak.ranges(min_start=0, max_start=DEFAULT_MAX_SIZE, max_end=100)
    )

    drawn = (
        ('min_size', min_size),
        ('max_size', max_size),
    )

    kwargs = draw(
        st.fixed_dictionaries(
            {k: st.just(v) for k, v in drawn if v is not None},
        )
    )

    return chain.extend(cast(BytestringContentsKwargs, kwargs))


@settings(max_examples=200)
@given(data=st.data())
def test_bytestring_contents(data: st.DataObject) -> None:
    '''Test that `bytestring_contents()` respects all its options.'''
    # Draw options
    opts = data.draw(bytestring_contents_kwargs(), label='opts')
    opts.reset()

    # Call the test subject
    result = data.draw(
        st_ak.contents.bytestring_contents(**opts.kwargs), label='result'
    )

    # Assert the result is always a ListOffsetArray
    assert isinstance(result, ListOffsetArray)

    # Assert outer __array__ parameter
    assert result.parameter('__array__') == 'bytestring'

    # Assert inner content is NumpyArray
    assert isinstance(result.content, NumpyArray)

    # Assert inner __array__ parameter
    assert result.content.parameter('__array__') == 'byte'

    # Assert inner dtype is uint8
    assert result.content.dtype == np.dtype(np.uint8)

    # Assert size bounds
    min_size = opts.kwargs.get('min_size', 0)
    max_size = opts.kwargs.get('max_size', DEFAULT_MAX_SIZE)

    assert sc(min_size) <= len(result) <= sc(max_size)

    # Assert all elements are bytes
    arr = ak.Array(result)
    for b in arr.to_list():
        assert isinstance(b, bytes)


def test_draw_empty() -> None:
    '''Assert that empty arrays (zero bytestrings) can be drawn.'''
    find(
        st_ak.contents.bytestring_contents(),
        lambda c: len(c) == 0,
        settings=settings(phases=[Phase.generate]),
    )


def test_draw_max_size() -> None:
    '''Assert that arrays with exactly max_size bytestrings can be drawn.'''
    find(
        st_ak.contents.bytestring_contents(),
        lambda c: len(c) == DEFAULT_MAX_SIZE,
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_empty_bytestring() -> None:
    '''Assert that arrays containing an empty bytestring can be drawn.'''
    find(
        st_ak.contents.bytestring_contents(min_size=1),
        lambda c: b'' in ak.Array(c).to_list(),
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_null_byte() -> None:
    '''Assert that arrays containing a bytestring with a null byte can be drawn.'''
    find(
        st_ak.contents.bytestring_contents(min_size=1),
        lambda c: any(b'\x00' in b for b in ak.Array(c).to_list()),
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )
