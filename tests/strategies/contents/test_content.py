from __future__ import annotations

from contextlib import ExitStack
from typing import Any, TypedDict, cast

import numpy as np
import pytest
from hypothesis import HealthCheck, find, given, settings
from hypothesis import strategies as st

import awkward as ak
from awkward.contents import EmptyArray
from hypothesis_awkward import strategies as st_ak
from hypothesis_awkward.util import (
    any_nan_nat_in_awkward_array,
    content_size,
    is_leaf,
    iter_contents,
    iter_numpy_arrays,
    leaf_size,
)
from hypothesis_awkward.util import safe_compare as sc

DEFAULT_MAX_SIZE = 50
DEFAULT_MAX_DEPTH = None


class ContentsKwargs(TypedDict, total=False):
    """Options for `contents()` strategy."""

    dtypes: st.SearchStrategy[np.dtype] | None
    max_size: int
    allow_nan: bool
    allow_numpy: bool
    allow_empty: bool
    allow_string: bool
    allow_bytestring: bool
    allow_regular: bool
    allow_list_offset: bool
    allow_list: bool
    allow_record: bool
    allow_union: bool
    allow_indexed_option: bool
    allow_byte_masked: bool
    allow_bit_masked: bool
    allow_unmasked: bool
    max_leaf_size: int | None
    max_depth: int | None
    min_length: int
    max_length: int | None


@st.composite
def contents_kwargs(
    draw: st.DrawFn,
    chain: st_ak.OptsChain[Any] | None = None,
) -> st_ak.OptsChain[ContentsKwargs]:
    """Strategy for options for `contents()` strategy."""
    if chain is None:
        chain = st_ak.OptsChain({})
    st_dtypes = chain.register(st_ak.supported_dtypes())

    min_length, max_length = draw(st_ak.ranges(min_start=0, max_end=10))

    drawn = (
        ('min_length', min_length),
        ('max_length', max_length),
    )

    kwargs = draw(
        st.fixed_dictionaries(
            {k: st.just(v) for k, v in drawn if v is not None},
            optional={
                'dtypes': st.one_of(
                    st.none(),
                    st.just(st_dtypes),
                ),
                'max_size': st.integers(min_value=0, max_value=200),
                'allow_nan': st.booleans(),
                'allow_numpy': st.booleans(),
                'allow_empty': st.booleans(),
                'allow_string': st.booleans(),
                'allow_bytestring': st.booleans(),
                'allow_regular': st.booleans(),
                'allow_list_offset': st.booleans(),
                'allow_list': st.booleans(),
                'allow_record': st.booleans(),
                'allow_union': st.booleans(),
                'allow_indexed_option': st.booleans(),
                'allow_byte_masked': st.booleans(),
                'allow_bit_masked': st.booleans(),
                'allow_unmasked': st.booleans(),
                'max_leaf_size': st.one_of(
                    st.none(), st.integers(min_value=0, max_value=50)
                ),
                'max_depth': st.one_of(
                    st.none(), st.integers(min_value=0, max_value=5)
                ),
            },
        )
    )

    return chain.extend(cast(ContentsKwargs, kwargs))


@settings(
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)
@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """Assert the results of `contents()`."""
    # Draw options
    opts = data.draw(contents_kwargs(), label='opts')
    opts.reset()

    # Assert that infeasible leaf-only configurations raise an error
    allow_numpy = opts.kwargs.get('allow_numpy', True)
    allow_empty = opts.kwargs.get('allow_empty', True)
    allow_string = opts.kwargs.get('allow_string', True)
    allow_bytestring = opts.kwargs.get('allow_bytestring', True)
    allow_regular = opts.kwargs.get('allow_regular', True)
    allow_list_offset = opts.kwargs.get('allow_list_offset', True)
    allow_list = opts.kwargs.get('allow_list', True)
    allow_record = opts.kwargs.get('allow_record', True)
    allow_union = opts.kwargs.get('allow_union', True)
    allow_indexed_option = opts.kwargs.get('allow_indexed_option', True)
    allow_byte_masked = opts.kwargs.get('allow_byte_masked', True)
    allow_bit_masked = opts.kwargs.get('allow_bit_masked', True)
    allow_unmasked = opts.kwargs.get('allow_unmasked', True)
    min_length = opts.kwargs.get('min_length', 0)
    max_depth = opts.kwargs.get('max_depth', DEFAULT_MAX_DEPTH)

    def _expect_raised() -> bool:
        if not (allow_numpy or allow_empty or allow_string or allow_bytestring):
            return True  # no leaves at all
        if min_length > 0:
            # Outermost is forced to be a leaf only when no wrapper/option is
            # allowed. In that case, EmptyArray is excluded by min_length>0,
            # and leaf_contents() raises if no other leaf type is allowed.
            # max_depth<=0 also short-circuits to the leaf-only path in
            # contents() (see content.py:220-223), making wrapper/option
            # allowances irrelevant at the outermost level.
            forced_leaf_only = max_depth is not None and max_depth <= 0
            no_outer_wrapper = forced_leaf_only or not any(
                (
                    allow_regular,
                    allow_list_offset,
                    allow_list,
                    allow_record,
                    allow_union,
                )
            )
            no_outer_option = forced_leaf_only or not any(
                (
                    allow_indexed_option,
                    allow_byte_masked,
                    allow_bit_masked,
                    allow_unmasked,
                )
            )
            if no_outer_wrapper and no_outer_option:
                if not (allow_numpy or allow_string or allow_bytestring):
                    return True
        return False

    # Call the test subject
    expect_raised = False
    with ExitStack() as stack:
        if _expect_raised():
            expect_raised = True
            stack.enter_context(pytest.raises(ValueError))
        c = data.draw(st_ak.contents.contents(**opts.kwargs), label='c')

    if expect_raised:
        return

    # Assert the result is always an ak.contents.Content
    assert isinstance(c, ak.contents.Content)

    # Assert the options were effective
    dtypes = opts.kwargs.get('dtypes', None)
    max_size = opts.kwargs.get('max_size', DEFAULT_MAX_SIZE)
    allow_nan = opts.kwargs.get('allow_nan', True)
    max_leaf_size = opts.kwargs.get('max_leaf_size')
    max_length = opts.kwargs.get('max_length')

    allow_any_nesting = any(
        (allow_regular, allow_list_offset, allow_list, allow_record, allow_union)
    )
    allow_any_option = any(
        (allow_indexed_option, allow_byte_masked, allow_bit_masked, allow_unmasked)
    )
    if not allow_any_nesting and not allow_any_option:
        assert _nesting_depth(c) == 0

    # Per-type gating
    if not allow_numpy:
        assert not _has_numpy(c)
    if not allow_empty:
        assert not _has_empty(c)
    if not allow_regular:
        assert not _has_regular(c)
    if not allow_list_offset:
        assert not _has_list_offset(c)
    if not allow_list:
        assert not _has_list(c)
    if not allow_string:
        assert not _has_string(c)
    if not allow_bytestring:
        assert not _has_bytestring(c)
    if not allow_record:
        assert not _has_record(c)
    if not allow_union:
        assert not _has_union(c)
    if not allow_indexed_option:
        assert not _has_indexed_option(c)
    if not allow_byte_masked:
        assert not _has_byte_masked(c)
    if not allow_bit_masked:
        assert not _has_bit_masked(c)
    if not allow_unmasked:
        assert not _has_unmasked(c)

    # Dtype check via leaf arrays (works for both flat and nested layouts)
    match dtypes:
        case None:
            pass
        case st_ak.RecordDraws():
            drawn_dtype_names = {d.name for d in dtypes.drawn}
            leaf_dtype_names = {d.name for d in _leaf_dtypes(c)}
            assert leaf_dtype_names <= drawn_dtype_names

    assert content_size(c) <= max_size

    if not allow_nan:
        assert not any_nan_nat_in_awkward_array(c)

    if max_leaf_size is not None:
        assert leaf_size(c) <= max_leaf_size
    assert _nesting_depth(c) <= sc(max_depth)
    assert min_length <= len(c) <= sc(max_length)


def test_draw_max_size() -> None:
    """Assert that content at exactly max_size can be drawn."""
    max_size = 30
    find(
        st_ak.contents.contents(max_size=max_size, max_leaf_size=max_size),
        lambda c: content_size(c) == max_size,
    )


def test_draw_max_leaf_size() -> None:
    """Assert that content at exactly max_leaf_size can be drawn."""
    max_leaf_size = 20
    find(
        st_ak.contents.contents(max_size=200, max_leaf_size=max_leaf_size),
        lambda c: leaf_size(c) == max_leaf_size,
    )


def test_draw_max_depth() -> None:
    """Assert that content at exactly max_depth can be drawn."""
    max_depth = 8
    find(
        st_ak.contents.contents(max_size=200, max_depth=max_depth),
        lambda c: _nesting_depth(c) == max_depth,
    )


def test_draw_deep_without_max_depth() -> None:
    """Assert that deep content can be drawn without specifying max_depth."""
    find(st_ak.contents.contents(max_size=200), lambda c: _nesting_depth(c) >= 8)


def test_draw_nested() -> None:
    """Assert that nested content (depth >= 2) can be drawn."""
    find(st_ak.contents.contents(max_leaf_size=20), lambda c: _nesting_depth(c) >= 2)


@pytest.mark.parametrize('min_length', [1, 2, 5])
@pytest.mark.parametrize('leaf', [True, False])
def test_draw_min_length(leaf: bool, min_length: int) -> None:
    """Assert that min_length constrains the content length."""
    find(
        st_ak.contents.contents(min_length=min_length),
        lambda c: is_leaf(c) == leaf and len(c) == min_length,
    )


@pytest.mark.parametrize('min_length', [1, 2, 5])
@pytest.mark.parametrize('leaf', [True, False])
def test_draw_min_length_not_recursed(leaf: bool, min_length: int) -> None:
    """Assert that min_length does not constrain nested content length."""
    # Inner contents may be shorter than min_length (the bound is
    # outer-only), mirroring the existing max_length-not-recursed test.
    find(
        st_ak.contents.contents(min_length=min_length),
        lambda c: any(
            len(n) < min_length and is_leaf(n) == leaf
            for n in iter_contents(c)
            if n is not c
        ),
    )


@pytest.mark.parametrize('max_length', [1, 2, 5])
@pytest.mark.parametrize('leaf', [True, False])
def test_draw_max_length(leaf: bool, max_length: int) -> None:
    """Assert that max_length constrains the content length."""
    find(
        st_ak.contents.contents(max_length=max_length),
        lambda c: is_leaf(c) == leaf and len(c) == max_length,
    )


@pytest.mark.parametrize('max_length', [1, 2, 5])
@pytest.mark.parametrize('leaf', [True, False])
def test_draw_max_length_not_recursed(leaf: bool, max_length: int) -> None:
    """Assert that max_length does not constrain nested content length."""
    # TODO: Pass a reachable-only option to `iter_contents` once supported,
    # so the predicate counts only genuinely reachable inner contents. See
    # the matching TODO in src/hypothesis_awkward/util/awkward/iter.py.
    find(
        st_ak.contents.contents(max_length=max_length),
        lambda c: any(
            len(n) > max_length and is_leaf(n) == leaf
            for n in iter_contents(c)
            if n is not c
        ),
    )


def _leaf_dtypes(c: ak.contents.Content) -> set[np.dtype]:
    """Dtypes of leaf NumPy arrays contained in `c`."""
    return {arr.dtype for arr in iter_numpy_arrays(c)}


def _nesting_depth(c: ak.contents.Content) -> int:
    """Maximum structural nesting depth (deepest path through the tree)."""
    _nesting_types = (
        ak.contents.RegularArray,
        ak.contents.ListOffsetArray,
        ak.contents.ListArray,
    )
    _option_types = (
        ak.contents.IndexedOptionArray,
        ak.contents.ByteMaskedArray,
        ak.contents.BitMaskedArray,
        ak.contents.UnmaskedArray,
    )
    if isinstance(c, _option_types):
        return _nesting_depth(c.content)
    if isinstance(c, ak.contents.UnionArray):
        return 1 + max(_nesting_depth(child) for child in c.contents)
    if isinstance(c, ak.contents.RecordArray):
        if not c.contents:
            return 1
        return 1 + max(_nesting_depth(child) for child in c.contents)
    if isinstance(c, _nesting_types):
        if c.parameter('__array__') in ('string', 'bytestring'):
            return 0
        return 1 + _nesting_depth(c.content)
    return 0


def _has_numpy(c: ak.contents.Content) -> bool:
    """Check if the content contains any NumpyArray node."""
    return any(isinstance(n, ak.contents.NumpyArray) for n in iter_contents(c))


def _has_empty(c: ak.contents.Content) -> bool:
    """Check if the content contains any EmptyArray node."""
    return any(isinstance(n, ak.contents.EmptyArray) for n in iter_contents(c))


def _has_list_offset(c: ak.contents.Content) -> bool:
    """Check if the content contains any structural ListOffsetArray node."""
    return any(
        isinstance(n, ak.contents.ListOffsetArray)
        and n.parameter('__array__') not in ('string', 'bytestring')
        for n in iter_contents(c)
    )


def _has_regular(c: ak.contents.Content) -> bool:
    """Check if the content contains any RegularArray node."""
    return any(isinstance(n, ak.contents.RegularArray) for n in iter_contents(c))


def _has_list(c: ak.contents.Content) -> bool:
    """Check if the content contains any ListArray node."""
    return any(isinstance(n, ak.contents.ListArray) for n in iter_contents(c))


def _has_string(c: ak.contents.Content) -> bool:
    """Check if the content contains any string node."""
    return any(n.parameter('__array__') == 'string' for n in iter_contents(c))


def _has_bytestring(c: ak.contents.Content) -> bool:
    """Check if the content contains any bytestring node."""
    return any(n.parameter('__array__') == 'bytestring' for n in iter_contents(c))


def _has_record(c: ak.contents.Content) -> bool:
    """Check if the content contains any RecordArray node."""
    return any(isinstance(n, ak.contents.RecordArray) for n in iter_contents(c))


def _has_union(c: ak.contents.Content) -> bool:
    """Check if the content contains any UnionArray node."""
    return any(isinstance(n, ak.contents.UnionArray) for n in iter_contents(c))


def _has_indexed_option(c: ak.contents.Content) -> bool:
    """Check if the content contains any IndexedOptionArray node."""
    return any(isinstance(n, ak.contents.IndexedOptionArray) for n in iter_contents(c))


def _has_byte_masked(c: ak.contents.Content) -> bool:
    """Check if the content contains any ByteMaskedArray node."""
    return any(isinstance(n, ak.contents.ByteMaskedArray) for n in iter_contents(c))


def _has_bit_masked(c: ak.contents.Content) -> bool:
    """Check if the content contains any BitMaskedArray node."""
    return any(isinstance(n, ak.contents.BitMaskedArray) for n in iter_contents(c))


def _has_unmasked(c: ak.contents.Content) -> bool:
    """Check if the content contains any UnmaskedArray node."""
    return any(isinstance(n, ak.contents.UnmaskedArray) for n in iter_contents(c))


def test_draw_from_contents_indexed_option() -> None:
    """Assert that IndexedOptionArray can be drawn from `contents()`."""
    find(st_ak.contents.contents(), _has_indexed_option)


def test_draw_from_contents_byte_masked() -> None:
    """Assert that ByteMaskedArray can be drawn from `contents()`."""
    find(st_ak.contents.contents(), _has_byte_masked)


def test_draw_from_contents_bit_masked() -> None:
    """Assert that BitMaskedArray can be drawn from `contents()`."""
    find(st_ak.contents.contents(), _has_bit_masked)


def test_draw_from_contents_unmasked() -> None:
    """Assert that UnmaskedArray can be drawn from `contents()`."""
    find(st_ak.contents.contents(), _has_unmasked)


def test_shrink_len_zero() -> None:
    """Assert that length-zero shrinks to `EmptyArray`."""
    c = find(
        st_ak.contents.contents(),
        lambda c: len(c) == 0,
        settings=settings(database=None),
    )
    assert isinstance(c, EmptyArray)


def test_shrink_len_positive() -> None:
    """Assert that length-positive shrinks."""
    c = find(
        st_ak.contents.contents(),
        lambda c: len(c) > 0,
        settings=settings(database=None),
    )
    assert len(c) == 1
    assert content_size(c) <= 2
