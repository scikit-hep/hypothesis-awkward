from typing import TypedDict, cast

import numpy as np
from hypothesis import find, given
from hypothesis import strategies as st

import awkward as ak
from awkward.contents import NumpyArray
from hypothesis_awkward import strategies as st_ak
from hypothesis_awkward.util import (
    any_nan_in_numpy_array,
    any_nan_nat_in_numpy_array,
    any_nat_in_numpy_array,
)
from tests.find_settings import FIND
from tests.funcs import KwargsDrawn, assert_kwargs_match_signature


class NumpyArrayContentsKwargs(TypedDict, total=False):
    """Options for `numpy_array_contents()` strategy."""

    dtypes: st.SearchStrategy[np.dtype] | None
    allow_nan: bool
    min_size: int
    max_size: int


DEFAULTS = NumpyArrayContentsKwargs(
    dtypes=None, allow_nan=True, min_size=0, max_size=10
)


class NumpyArrayContentsDrawn(TypedDict, total=False):
    """Values drawn in advance for the options of `numpy_array_contents()`."""

    dtypes: np.dtype


class RelatedKwargs(TypedDict, total=False):
    """Options drawn together as an ordered pair (`min_size <= max_size`)."""

    min_size: int
    max_size: int


def test_kwargs_match_signature() -> None:
    """Assert the option declarations agree with `numpy_array_contents()`."""
    assert_kwargs_match_signature(
        func=st_ak.contents.numpy_array_contents,
        kwargs_cls=NumpyArrayContentsKwargs,
        defaults=DEFAULTS,
        related_cls=RelatedKwargs,
    )


@st.composite
def numpy_array_contents_kwargs(
    draw: st.DrawFn,
) -> KwargsDrawn[NumpyArrayContentsKwargs, NumpyArrayContentsDrawn]:
    """Strategy for options for `numpy_array_contents()` strategy."""

    @st.composite
    def _st_related_kwargs(draw: st.DrawFn) -> RelatedKwargs:
        """Strategy for the options that are drawn together."""
        min_size, max_size = draw(
            st_ak.ranges(min_start=0, max_start=DEFAULTS['max_size'], max_end=100)
        )
        ret = RelatedKwargs()
        if min_size is not None:
            ret['min_size'] = min_size
        if max_size is not None:
            ret['max_size'] = max_size
        return ret

    @st.composite
    def _st_drawn(draw: st.DrawFn) -> NumpyArrayContentsDrawn:
        """Strategy for the values drawn in advance for the options."""
        ret = NumpyArrayContentsDrawn()
        if draw(st.booleans()):
            ret['dtypes'] = draw(st_ak.supported_dtypes())
        return ret

    @st.composite
    def _st_optional_independent(
        draw: st.DrawFn, drawn: NumpyArrayContentsDrawn
    ) -> NumpyArrayContentsKwargs:
        """Strategy for the options that are drawn independently, if at all."""
        ret = cast(
            NumpyArrayContentsKwargs,
            draw(st.fixed_dictionaries({}, optional={'allow_nan': st.booleans()})),
        )
        if 'dtypes' in drawn:
            ret['dtypes'] = st.just(drawn['dtypes'])
        elif draw(st.booleans()):
            ret['dtypes'] = None
        return ret

    related = draw(_st_related_kwargs())
    drawn = draw(_st_drawn())
    optional_independent = draw(_st_optional_independent(drawn))

    kwargs = NumpyArrayContentsKwargs(**related, **optional_independent)
    return KwargsDrawn(kwargs, drawn)


@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """Assert the results of `numpy_array_contents()`."""
    # Draw options
    kwargs, drawn = data.draw(numpy_array_contents_kwargs(), label='opts')

    # Call the test subject
    result = data.draw(st_ak.contents.numpy_array_contents(**kwargs), label='result')

    # Assert the result is always a NumpyArray content
    assert isinstance(result, ak.contents.NumpyArray)

    # Assert underlying data is 1-D (from max_dims=1)
    assert result.data.ndim == 1

    # Assert not structured (from allow_structured=False)
    assert result.data.dtype.names is None

    # Assert size bounds
    allow_nan = kwargs.get('allow_nan', DEFAULTS['allow_nan'])
    min_size = kwargs.get('min_size', DEFAULTS['min_size'])
    max_size = kwargs.get('max_size', DEFAULTS['max_size'])

    assert min_size <= len(result) <= max_size

    # Assert allow_nan
    if not allow_nan:
        assert not any_nan_nat_in_numpy_array(result.data)

    # Assert dtypes
    if 'dtypes' in drawn:
        assert result.data.dtype == drawn['dtypes']


def test_draw_from_contents() -> None:
    """Assert `contents()` can generate a `NumpyArray` as outermost."""
    find(st_ak.contents.contents(), lambda c: isinstance(c, NumpyArray), settings=FIND)


def test_draw_from_contents_length_zero() -> None:
    """Assert that NumpyArray with length 0 can be drawn from `contents()`."""
    find(
        st_ak.contents.contents(),
        lambda c: isinstance(c, NumpyArray) and len(c) == 0,
        settings=FIND,
    )


def test_draw_from_contents_max_size_1() -> None:
    """Assert that a NumpyArray with one element can be drawn with max_size=1."""
    find(
        st_ak.contents.contents(max_size=1),
        lambda c: isinstance(c, NumpyArray) and len(c) == 1,
        settings=FIND,
    )


def test_draw_empty() -> None:
    """Assert that empty arrays can be drawn."""
    find(st_ak.contents.numpy_array_contents(), lambda c: len(c) == 0, settings=FIND)


def test_draw_max_size() -> None:
    """Assert that arrays with exactly max_size elements can be drawn."""
    find(
        st_ak.contents.numpy_array_contents(),
        lambda c: len(c) == DEFAULTS['max_size'],
        settings=FIND,
    )


def test_draw_min_size() -> None:
    """Assert that arrays with exactly min_size elements can be drawn."""
    min_size = 5
    find(
        st_ak.contents.numpy_array_contents(min_size=min_size),
        lambda c: len(c) == min_size,
        settings=FIND,
    )


def test_draw_nan() -> None:
    """Assert that arrays with NaN can be drawn when allowed."""
    float_dtypes = st_ak.supported_dtypes().filter(lambda d: d.kind == 'f')
    find(
        st_ak.contents.numpy_array_contents(
            dtypes=float_dtypes, allow_nan=True, min_size=1
        ),
        lambda c: any_nan_in_numpy_array(c.data),
        settings=FIND,
    )


def test_draw_nat() -> None:
    """Assert that arrays with NaT can be drawn when allowed."""
    datetime_dtypes = st_ak.supported_dtypes().filter(lambda d: d.kind in ('M', 'm'))
    find(
        st_ak.contents.numpy_array_contents(
            dtypes=datetime_dtypes, allow_nan=True, min_size=1
        ),
        lambda c: any_nat_in_numpy_array(c.data),
        settings=FIND,
    )
