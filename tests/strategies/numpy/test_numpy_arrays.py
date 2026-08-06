import math
from typing import Any, TypedDict, cast

import numpy as np
import pytest
from hypothesis import find, given
from hypothesis import strategies as st
from hypothesis.extra import numpy as st_np

import awkward as ak
from hypothesis_awkward import strategies as st_ak
from hypothesis_awkward.util import (
    any_nan_in_numpy_array,
    any_nan_nat_in_numpy_array,
    any_nat_in_numpy_array,
    n_scalars_in,
    simple_dtype_kinds_in,
)
from hypothesis_awkward.util import safe_compare as sc
from tests.find_settings import FIND_NO_SHRINK
from tests.funcs import assert_kwargs_match_signature


class NumpyArraysKwargs(TypedDict, total=False):
    """Options for `numpy_arrays()` strategy."""

    dtype: np.dtype | st.SearchStrategy[np.dtype] | None
    allow_structured: bool
    allow_nan: bool
    min_dims: int
    max_dims: int | None
    min_size: int
    max_size: int
    unique: bool


DEFAULTS = NumpyArraysKwargs(
    dtype=None,
    allow_structured=True,
    allow_nan=True,
    min_dims=1,
    max_dims=None,
    min_size=0,
    max_size=10,
    unique=False,
)


class RelatedKwargs(TypedDict, total=False):
    """Options drawn together: the dtype pool depends on the other options.

    `bool` dtypes are excluded when `unique` is drawn and `min_size` may exceed the
    two distinct `bool` values.
    """

    min_dims: int
    max_dims: int | None
    min_size: int
    max_size: int
    unique: bool
    dtype: np.dtype | st.SearchStrategy[np.dtype] | None


def test_kwargs_match_signature() -> None:
    """Assert the option declarations agree with `numpy_arrays()`'s parameters."""
    assert_kwargs_match_signature(
        func=st_ak.numpy_arrays,
        kwargs_cls=NumpyArraysKwargs,
        defaults=DEFAULTS,
        related_cls=RelatedKwargs,
    )


@st.composite
def numpy_arrays_kwargs(
    draw: st.DrawFn,
    chain: st_ak.OptsChain[Any] | None = None,
) -> st_ak.OptsChain[NumpyArraysKwargs]:
    """Strategy for options for `numpy_arrays()` strategy."""
    if chain is None:
        chain = st_ak.OptsChain({})

    @st.composite
    def _st_related_kwargs(draw: st.DrawFn) -> RelatedKwargs:
        """Strategy for the options that are drawn together."""
        min_dims, max_dims = draw(st_ak.ranges(min_start=1, max_end=5))
        min_size, max_size = draw(
            st_ak.ranges(min_start=0, max_end=100, max_start=DEFAULTS['max_size'])
        )
        unique = draw(st_ak.none_or(st.booleans()))

        st_dtypes = st_ak.supported_dtypes()
        if unique and not sc(min_size) <= 2:
            st_dtypes = st_dtypes.filter(lambda d: d.kind != 'b')
        registered_st_dtypes = chain.register(st_dtypes)
        dtype = draw(st.one_of(st.none(), st.just(registered_st_dtypes), st_dtypes))

        ret = RelatedKwargs()
        if min_dims is not None:
            ret['min_dims'] = min_dims
        if max_dims is not None:
            ret['max_dims'] = max_dims
        if min_size is not None:
            ret['min_size'] = min_size
        if max_size is not None:
            ret['max_size'] = max_size
        if unique is not None:
            ret['unique'] = unique
        if dtype is not None:
            ret['dtype'] = dtype
        return ret

    related = draw(_st_related_kwargs())

    optional_independent = draw(
        st.fixed_dictionaries(
            {},
            optional={
                'allow_structured': st.booleans(),
                'allow_nan': st.booleans(),
            },
        )
    )

    return chain.extend(cast(NumpyArraysKwargs, {**related, **optional_independent}))


def _all_distinct(a: np.ndarray) -> bool:
    """Return True if all elements of `a` are pairwise distinct.

    Compares NumPy scalars (not `a.tolist()`) so that `NaN`/`NaT` count as distinct,
    matching how Hypothesis generates `unique=True` arrays. `a.tolist()` maps `NaT` to
    `None`, and `None == None` is `True`, which would collapse distinct `NaT`s.
    """
    vals = list(a.ravel())
    return all(
        not bool(vals[i] == vals[j])
        for i in range(len(vals))
        for j in range(i + 1, len(vals))
    )


@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """Assert the results of `numpy_arrays()`."""
    # Draw options
    opts = data.draw(numpy_arrays_kwargs(), label='opts')
    opts.reset()

    # Call the test subject
    n = data.draw(st_ak.numpy_arrays(**opts.kwargs), label='n')

    # Assert the options were effective
    dtype = opts.kwargs.get('dtype', DEFAULTS['dtype'])
    allow_structured = opts.kwargs.get('allow_structured', DEFAULTS['allow_structured'])
    allow_nan = opts.kwargs.get('allow_nan', DEFAULTS['allow_nan'])
    min_dims = opts.kwargs.get('min_dims', DEFAULTS['min_dims'])
    max_dims = opts.kwargs.get('max_dims', DEFAULTS['max_dims'])
    min_size = opts.kwargs.get('min_size', DEFAULTS['min_size'])
    max_size = opts.kwargs.get('max_size', DEFAULTS['max_size'])
    unique = opts.kwargs.get('unique', DEFAULTS['unique'])

    match dtype:
        case np.dtype():
            kinds = simple_dtype_kinds_in(n.dtype)
            assert len(kinds) == 1
            assert dtype.kind in kinds
        case st_ak.RecordDraws():
            drawn_kinds = {d.kind for d in dtype.drawn}
            result_kinds = simple_dtype_kinds_in(n.dtype)
            assert result_kinds <= drawn_kinds

    n_scalars = math.prod(n.shape) * n_scalars_in(n.dtype)
    assert min_size <= n_scalars <= max_size

    structured = n.dtype.names is not None
    has_nan = any_nan_nat_in_numpy_array(n)

    if not allow_structured:
        assert not structured

    if not allow_nan:
        assert not has_nan

    assert min_dims <= len(n.shape) <= sc(max_dims)

    if unique:
        assert _all_distinct(n)

    # Assert an Awkward Array can be created.
    a = ak.from_numpy(n)
    assert isinstance(a, ak.Array)

    # Test if the NumPy array and Awkward Array are converted to the same list.
    # Compare only when `NaN` isn't allowed.
    # Structured arrays are known to result in a different list sometimes.
    to_list = a.to_list()

    if not allow_nan:
        if not structured:  # simple array
            assert to_list == n.tolist()
        else:  # structured array
            # assert to_list == n.tolist()  # NOTE: Fails sometimes
            pass

    # Test if the Awkward Array is converted back to a NumPy array with the identical
    # values. The conversion of structured arrays fails under a known condition.
    # Structured arrays may not result in identical values.

    def _is_numpy_convertible(a: ak.Array) -> bool:
        """True if `a.to_numpy()` is expected to work without error.

        `to_numpy()` fails for structured arrays with non-1D fields
        https://github.com/scikit-hep/awkward/issues/3690
        """
        layout = a.layout
        if isinstance(layout, ak.contents.NumpyArray):  # simple array
            return True
        assert isinstance(layout, ak.contents.RecordArray)  # structured array
        return all(len(c.shape) == 1 for c in layout.contents)

    if _is_numpy_convertible(a):
        to_numpy = a.to_numpy()
        if not has_nan:
            if not structured:
                np.testing.assert_array_equal(to_numpy, n)
            else:
                # np.testing.assert_array_equal(to_numpy, n)  # NOTE: Fails sometimes
                pass
    else:
        with pytest.raises(ValueError):
            a.to_numpy()


def test_draw_structured() -> None:
    """Assert that structured arrays can be drawn by default."""
    find(
        st_ak.numpy_arrays(),
        lambda a: a.dtype.names is not None,
        settings=FIND_NO_SHRINK,
    )


def test_draw_nan() -> None:
    """Assert that arrays with NaN can be drawn when allowed."""
    find(
        st_ak.numpy_arrays(dtype=st_np.floating_dtypes(), allow_nan=True),
        lambda a: any_nan_in_numpy_array(a),
        settings=FIND_NO_SHRINK,
    )


def test_draw_nat_datetime64() -> None:
    """Assert that datetime64 arrays with NaT can be drawn when allowed."""
    find(
        st_ak.numpy_arrays(dtype=st_np.datetime64_dtypes(), allow_nan=True),
        lambda a: any_nat_in_numpy_array(a),
        settings=FIND_NO_SHRINK,
    )


def test_draw_nat_timedelta64() -> None:
    """Assert that timedelta64 arrays with NaT can be drawn when allowed."""
    find(
        st_ak.numpy_arrays(dtype=st_np.timedelta64_dtypes(), allow_nan=True),
        lambda a: any_nat_in_numpy_array(a),
        settings=FIND_NO_SHRINK,
    )


def test_draw_empty() -> None:
    """Assert that empty arrays can be drawn by default."""
    find(
        st_ak.numpy_arrays(), lambda a: math.prod(a.shape) == 0, settings=FIND_NO_SHRINK
    )


@pytest.mark.parametrize('max_dims', [1, None])
@pytest.mark.parametrize('max_size', [1, DEFAULTS['max_size']])
@pytest.mark.parametrize('allow_structured', [True, False])
def test_draw_empty_parametrized(
    max_size: int, max_dims: int | None, allow_structured: bool
) -> None:
    """Assert that empty arrays can be drawn when max allows at most one scalar."""
    find(
        st_ak.numpy_arrays(
            max_dims=max_dims, max_size=max_size, allow_structured=allow_structured
        ),
        lambda a: math.prod(a.shape) == 0,
        settings=FIND_NO_SHRINK,
    )


def test_draw_max_size() -> None:
    """Assert that arrays with exactly max_size scalars can be drawn."""
    find(
        st_ak.numpy_arrays(allow_structured=False),
        lambda a: math.prod(a.shape) == DEFAULTS['max_size'],
        settings=FIND_NO_SHRINK,
    )


def test_draw_max_size_structured() -> None:
    """Assert that max_size counts scalars for structured dtypes."""
    find(
        st_ak.numpy_arrays(),
        lambda a: (
            math.prod(a.shape) * n_scalars_in(a.dtype) == DEFAULTS['max_size']
            and a.dtype.names is not None
        ),
        settings=FIND_NO_SHRINK,
    )


def test_draw_nonempty_max_size_1() -> None:
    """Assert that a non-empty array can be drawn with max_size=1."""
    find(
        st_ak.numpy_arrays(allow_structured=False, max_size=1),
        lambda a: math.prod(a.shape) == 1,
        settings=FIND_NO_SHRINK,
    )


def test_draw_min_size() -> None:
    """Assert that arrays with exactly min_size scalars can be drawn."""
    min_size = 5
    find(
        st_ak.numpy_arrays(allow_structured=False, min_size=min_size),
        lambda a: math.prod(a.shape) == min_size,
        settings=FIND_NO_SHRINK,
    )


def test_draw_min_size_structured() -> None:
    """Assert that min_size counts scalars for structured dtypes."""
    min_size = 5
    find(
        st_ak.numpy_arrays(min_size=min_size),
        lambda a: (
            math.prod(a.shape) * n_scalars_in(a.dtype) == min_size
            and a.dtype.names is not None
        ),
        settings=FIND_NO_SHRINK,
    )


def test_draw_one_dim() -> None:
    """Assert that 1-D arrays can be drawn by default."""
    find(
        st_ak.numpy_arrays(allow_structured=False),
        lambda a: len(a.shape) == 1,
        settings=FIND_NO_SHRINK,
    )


def test_draw_min_dims() -> None:
    """Assert that arrays with at least min_dims dimensions can be drawn."""
    find(
        st_ak.numpy_arrays(allow_structured=False, min_dims=2),
        lambda a: len(a.shape) == 2,
        settings=FIND_NO_SHRINK,
    )


def test_draw_max_dims() -> None:
    """Assert that arrays with max_dims dimensions can be drawn."""
    find(
        st_ak.numpy_arrays(allow_structured=False, max_dims=3),
        lambda a: len(a.shape) == 3,
        settings=FIND_NO_SHRINK,
    )


def test_draw_unique_bool() -> None:
    """Assert that [True, False] can be drawn when unique is True.

    The `dtype.kind == 'b'` guard is required because `{0, 1} == {True, False}` in
    Python, so a unique int array `[0, 1]` would otherwise match.
    """
    find(
        st_ak.numpy_arrays(unique=True),
        lambda a: a.dtype.kind == 'b' and set(a.ravel().tolist()) == {True, False},
        settings=FIND_NO_SHRINK,
    )
