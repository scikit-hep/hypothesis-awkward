from collections.abc import Iterator
from typing import Union

import numpy as np

import awkward as ak
from awkward.contents import (
    Content,
    EmptyArray,
    IndexedOptionArray,
    ListArray,
    ListOffsetArray,
    NumpyArray,
    RecordArray,
    RegularArray,
    UnionArray,
    UnmaskedArray,
)

LeafContent = Union[NumpyArray, EmptyArray, ListOffsetArray, ListArray, RegularArray]


def _is_string_or_bytestring_leaf(
    c: Content,
    string_as_leaf: bool,
    bytestring_as_leaf: bool,
) -> bool:
    array_param = c.parameter('__array__')
    if string_as_leaf and array_param == 'string':
        return True
    if bytestring_as_leaf and array_param == 'bytestring':
        return True
    return False


def any_nan_nat_in_awkward_array(a: ak.Array | Content, /) -> bool:
    '''`True` if Awkward Array contains any `NaN` or `NaT` values, else `False`.

    Parameters
    ----------
    a
        An Awkward Array.

    Returns
    -------
    bool
        `True` if `a` contains any `NaN` or `NaT` values, else `False`.

    Examples
    --------

    >>> a = ak.Array([1.0, 2.0, np.nan])
    >>> any_nan_nat_in_awkward_array(a)
    True

    >>> a = ak.Array([1.0, 2.0, 3.0])
    >>> any_nan_nat_in_awkward_array(a)
    False

    >>> a = ak.Array([{'x': 1.0, 'y': np.nan}, {'x': 2.0, 'y': 3.0}])
    >>> any_nan_nat_in_awkward_array(a)
    True

    '''
    return any_nan_in_awkward_array(a) or any_nat_in_awkward_array(a)


def any_nan_in_awkward_array(a: ak.Array | Content, /) -> bool:
    '''`True` if Awkward Array contains any `NaN` values, else `False`.

    Parameters
    ----------
    a
        An Awkward Array.

    Returns
    -------
    bool
        `True` if `a` contains any `NaN` values, else `False`.

    Examples
    --------

    >>> a = ak.Array([1.0, 2.0, np.nan])
    >>> any_nan_in_awkward_array(a)
    True

    >>> a = ak.Array([1.0, 2.0, 3.0])
    >>> any_nan_in_awkward_array(a)
    False

    >>> a = ak.Array([{'x': 1.0, 'y': np.nan}, {'x': 2.0, 'y': 3.0}])
    >>> any_nan_in_awkward_array(a)
    True

    '''
    for arr in iter_numpy_arrays(a):
        if arr.dtype.kind in {'f', 'c'} and np.any(np.isnan(arr)):
            return True
    return False


def any_nat_in_awkward_array(a: ak.Array | Content, /) -> bool:
    '''`True` if Awkward Array contains any `NaT` values, else `False`.

    Parameters
    ----------
    a
        An Awkward Array.

    Returns
    -------
    bool
        `True` if `a` contains any `NaT` values, else `False`.

    Examples
    --------

    >>> a = ak.Array(np.array(['2020-01-01', 'NaT'], dtype='datetime64[D]'))
    >>> any_nat_in_awkward_array(a)
    True

    >>> a = ak.Array(np.array(['2020-01-01', '2020-01-02'], dtype='datetime64[D]'))
    >>> any_nat_in_awkward_array(a)
    False

    '''
    for arr in iter_numpy_arrays(a):
        if arr.dtype.kind in {'m', 'M'} and np.any(np.isnat(arr)):
            return True
    return False


def iter_contents(
    a: ak.Array | Content,
    /,
    *,
    string_as_leaf: bool = True,
    bytestring_as_leaf: bool = True,
) -> Iterator[Content]:
    '''Iterate over all contents in an Awkward Array layout.

    Parameters
    ----------
    a
        An Awkward Array or Content.
    string_as_leaf
        If `True` (default), treat string `ListOffsetArray`/`ListArray`/
        `RegularArray` nodes as leaves — do not descend into the inner
        `NumpyArray(uint8)`.
    bytestring_as_leaf
        If `True` (default), treat bytestring nodes as leaves.

    Yields
    ------
    Content
        Each content node in the layout.

    '''
    stack: list[ak.Array | Content] = [a]
    while stack:
        item = stack.pop()
        match item:
            case ak.Array():
                stack.append(item.layout)
            case NumpyArray() | EmptyArray():
                yield item
            case RecordArray():
                yield item
                stack.extend(item.contents)
            case ListArray() | ListOffsetArray() | RegularArray() if (
                _is_string_or_bytestring_leaf(item, string_as_leaf, bytestring_as_leaf)
            ):
                yield item
            case (
                IndexedOptionArray()
                | ListArray()
                | ListOffsetArray()
                | RegularArray()
                | UnmaskedArray()
            ):
                yield item
                stack.append(item.content)
            case UnionArray():
                yield item
                stack.extend(item.contents)
            case _:  # pragma: no cover
                raise TypeError(f'Unexpected content type: {type(item)}')


def iter_leaf_contents(
    a: ak.Array | Content,
    /,
    *,
    string_as_leaf: bool = True,
    bytestring_as_leaf: bool = True,
) -> Iterator[LeafContent]:
    '''Iterate over all leaf contents in an Awkward Array layout.

    Parameters
    ----------
    a
        An Awkward Array or Content.
    string_as_leaf
        If `True` (default), treat string `ListOffsetArray`/`ListArray`/
        `RegularArray` nodes as leaves.
    bytestring_as_leaf
        If `True` (default), treat bytestring nodes as leaves.

    Yields
    ------
    LeafContent
        Each leaf content in the layout.

    '''
    for content in iter_contents(
        a, string_as_leaf=string_as_leaf, bytestring_as_leaf=bytestring_as_leaf
    ):
        if isinstance(content, (EmptyArray, NumpyArray)):
            yield content
        elif isinstance(content, (ListOffsetArray, ListArray, RegularArray)):
            if _is_string_or_bytestring_leaf(
                content, string_as_leaf, bytestring_as_leaf
            ):
                yield content


def iter_numpy_arrays(
    a: ak.Array | Content,
    /,
    *,
    exclude_string: bool = True,
    exclude_bytestring: bool = True,
) -> Iterator[np.ndarray]:
    '''Iterate over all NumPy arrays in an Awkward Array layout.

    Parameters
    ----------
    a
        An Awkward Array or Content.
    exclude_string
        If `True` (default), exclude the inner `uint8` data of string nodes.
    exclude_bytestring
        If `True` (default), exclude the inner `uint8` data of bytestring nodes.

    Yields
    ------
    np.ndarray
        Each underlying NumPy array in the layout.

    Examples
    --------

    >>> a = ak.Array([[1.0, 2.0], [3.0]])
    >>> list(iter_numpy_arrays(a))
    [array([1., 2., 3.])]

    >>> a = ak.Array([{'x': 1, 'y': 2.0}, {'x': 3, 'y': 4.0}])
    >>> sorted([arr.dtype for arr in iter_numpy_arrays(a)], key=str)
    [dtype('float64'), dtype('int64')]

    '''
    for content in iter_leaf_contents(
        a,
        string_as_leaf=exclude_string,
        bytestring_as_leaf=exclude_bytestring,
    ):
        if isinstance(content, NumpyArray):
            yield content.data
