from __future__ import annotations

from typing import Any, TypedDict, cast

import numpy as np
import pytest
from hypothesis import Phase, find, given, settings
from hypothesis import strategies as st

import awkward as ak
from awkward.contents import EmptyArray
from hypothesis_awkward import strategies as st_ak
from hypothesis_awkward.util import (
    any_nan_nat_in_awkward_array,
    content_size,
    is_bytestring_leaf,
    is_leaf,
    is_string_leaf,
    iter_contents,
    iter_numpy_arrays,
    leaf_size,
    safe_max,
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
    allow_indexed: bool
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
    max_size = draw(st.integers(min_value=safe_max([min_length, 0]), max_value=200))

    non_empty_leaves = {'numpy', 'string', 'bytestring'}
    all_leaves = non_empty_leaves | {'empty'}
    if min_length:
        # At least one non-empty leaf must be allowed.
        allowed = draw(
            st.sets(
                st.sampled_from(list(non_empty_leaves)),
                min_size=1,
                max_size=len(non_empty_leaves),
            )
        )
        if draw(st.booleans()):
            allowed.add('empty')
    else:
        # The length zero is possible.
        # At least any one leaf must be allowed.
        allowed = draw(
            st.sets(
                st.sampled_from(list(all_leaves)),
                min_size=1,
                max_size=len(all_leaves),
            )
        )

    drawn = (
        ('min_length', min_length),
        ('max_length', max_length),
        ('max_size', max_size),
        ('allow_numpy', 'numpy' in allowed),
        ('allow_empty', 'empty' in allowed),
        ('allow_string', 'string' in allowed),
        ('allow_bytestring', 'bytestring' in allowed),
    )

    kwargs = draw(
        st.fixed_dictionaries(
            {k: st.just(v) for k, v in drawn if v is not None},
            optional={
                'dtypes': st.just(st_dtypes),
                'allow_nan': st.booleans(),
                'allow_regular': st.booleans(),
                'allow_list_offset': st.booleans(),
                'allow_list': st.booleans(),
                'allow_record': st.booleans(),
                'allow_union': st.booleans(),
                'allow_indexed': st.booleans(),
                'allow_indexed_option': st.booleans(),
                'allow_byte_masked': st.booleans(),
                'allow_bit_masked': st.booleans(),
                'allow_unmasked': st.booleans(),
                'max_leaf_size': st.integers(min_value=0, max_value=50),
                'max_depth': st.integers(min_value=0, max_value=5),
            },
        )
    )

    return chain.extend(cast(ContentsKwargs, kwargs))


@settings(max_examples=200)
@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """Assert the results of `contents()`."""
    # Draw options
    opts = data.draw(contents_kwargs(), label='opts')
    opts.reset()

    # Call the test subject
    c = data.draw(st_ak.contents.contents(**opts.kwargs), label='c')

    # Assert the result is always an ak.contents.Content
    assert isinstance(c, ak.contents.Content)

    # Assert the options were effective
    dtypes = opts.kwargs.get('dtypes', None)
    max_size = opts.kwargs.get('max_size', DEFAULT_MAX_SIZE)
    allow_nan = opts.kwargs.get('allow_nan', True)
    allow_numpy = opts.kwargs.get('allow_numpy', True)
    allow_empty = opts.kwargs.get('allow_empty', True)
    allow_string = opts.kwargs.get('allow_string', True)
    allow_bytestring = opts.kwargs.get('allow_bytestring', True)
    allow_regular = opts.kwargs.get('allow_regular', True)
    allow_list_offset = opts.kwargs.get('allow_list_offset', True)
    allow_list = opts.kwargs.get('allow_list', True)
    allow_record = opts.kwargs.get('allow_record', True)
    allow_union = opts.kwargs.get('allow_union', True)
    allow_indexed = opts.kwargs.get('allow_indexed', True)
    allow_indexed_option = opts.kwargs.get('allow_indexed_option', True)
    allow_byte_masked = opts.kwargs.get('allow_byte_masked', True)
    allow_bit_masked = opts.kwargs.get('allow_bit_masked', True)
    allow_unmasked = opts.kwargs.get('allow_unmasked', True)
    max_leaf_size = opts.kwargs.get('max_leaf_size')
    min_length = opts.kwargs.get('min_length', 0)
    max_length = opts.kwargs.get('max_length')
    max_depth = opts.kwargs.get('max_depth', DEFAULT_MAX_DEPTH)

    allow_any_nesting = any(
        (allow_regular, allow_list_offset, allow_list, allow_record, allow_union)
    )
    allow_any_option = any(
        (allow_indexed_option, allow_byte_masked, allow_bit_masked, allow_unmasked)
    )
    if not allow_any_nesting and not allow_any_option:
        assert _nesting_depth(c) == 0

    # Per-type gating
    allow_by_type = {
        'numpy': allow_numpy,
        'empty': allow_empty,
        'regular': allow_regular,
        'list_offset': allow_list_offset,
        'list': allow_list,
        'string': allow_string,
        'bytestring': allow_bytestring,
        'record': allow_record,
        'union': allow_union,
        'indexed': allow_indexed,
        'indexed_option': allow_indexed_option,
        'byte_masked': allow_byte_masked,
        'bit_masked': allow_bit_masked,
        'unmasked': allow_unmasked,
    }
    present = _present_types(c)
    disallowed = {t for t, allowed in allow_by_type.items() if not allowed}
    assert not (present & disallowed), (
        f'disallowed types present: {sorted(present & disallowed)}'
    )

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


def test_draw_min_length_gt_leaf_max_size() -> None:
    """Assert that `contents()` does not raise when `min_length > max_leaf_size`.

    Regression test. When an option wrapper that forwards `min_length`
    (e.g. byte/bit-masked or unmasked) recurses with `allow_option_root=False`
    and `allow_union_root=False`, the inner call can have empty `candidates`
    while `leaf_max_size < min_length`. The fallback must `assume(...)` the
    leaf bound rather than raise `InvalidArgument`.
    """
    find(
        st_ak.contents.contents(
            max_size=3,
            max_leaf_size=1,
            min_length=2,
            allow_numpy=False,
            allow_empty=False,
            allow_bytestring=False,
            allow_regular=False,
            allow_list_offset=False,
            allow_list=False,
            allow_record=False,
            allow_bit_masked=False,
            allow_unmasked=False,
        ),
        lambda _: True,
        settings=settings(phases=[Phase.generate], derandomize=True),
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
    if isinstance(c, ak.contents.IndexedArray):
        return _nesting_depth(c.content)
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


def _present_types(c: ak.contents.Content) -> set[str]:
    """Collect the gated layout-type tags present anywhere in `c`."""
    present = set[str]()
    for n in iter_contents(c):
        if is_string_leaf(n):
            present.add('string')
            continue
        elif is_bytestring_leaf(n):
            present.add('bytestring')
            continue
        match n:
            case ak.contents.NumpyArray():
                present.add('numpy')
            case ak.contents.EmptyArray():
                present.add('empty')
            case ak.contents.RegularArray():
                present.add('regular')
            case ak.contents.ListOffsetArray():
                present.add('list_offset')
            case ak.contents.ListArray():
                present.add('list')
            case ak.contents.RecordArray():
                present.add('record')
            case ak.contents.UnionArray():
                present.add('union')
            case ak.contents.IndexedArray():
                present.add('indexed')
            case ak.contents.IndexedOptionArray():
                present.add('indexed_option')
            case ak.contents.ByteMaskedArray():
                present.add('byte_masked')
            case ak.contents.BitMaskedArray():
                present.add('bit_masked')
            case ak.contents.UnmaskedArray():
                present.add('unmasked')
    return present


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
