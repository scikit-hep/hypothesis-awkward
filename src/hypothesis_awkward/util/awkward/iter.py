from collections.abc import Iterator
from typing import Union

import numpy as np

import awkward as ak
from awkward.contents import (
    Content,
    EmptyArray,
    ListArray,
    ListOffsetArray,
    NumpyArray,
    RegularArray,
)

from .contents import get_contents
from .leaf import is_string_or_bytestring_leaf

LeafContent = Union[NumpyArray, EmptyArray, ListOffsetArray, ListArray, RegularArray]


def iter_contents(
    a: ak.Array | Content,
    /,
    *,
    string_as_leaf: bool = True,
    bytestring_as_leaf: bool = True,
) -> Iterator[Content]:
    """Iterate over all contents in an Awkward Array layout.

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
    """
    stack = list[Content]()
    if isinstance(a, ak.Array):
        stack.append(a.layout)
    else:
        stack.append(a)
    while stack:
        item = stack.pop()
        yield item
        contents = get_contents(
            item,
            string_as_leaf=string_as_leaf,
            bytestring_as_leaf=bytestring_as_leaf,
        )
        stack.extend(contents)


def iter_leaf_contents(
    a: ak.Array | Content,
    /,
    *,
    string_as_leaf: bool = True,
    bytestring_as_leaf: bool = True,
) -> Iterator[LeafContent]:
    """Iterate over all leaf contents in an Awkward Array layout.

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
    """
    for content in iter_contents(
        a, string_as_leaf=string_as_leaf, bytestring_as_leaf=bytestring_as_leaf
    ):
        if isinstance(content, (EmptyArray, NumpyArray)):
            yield content
        elif isinstance(content, (ListOffsetArray, ListArray, RegularArray)):
            if is_string_or_bytestring_leaf(
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
    """Iterate over all NumPy arrays in an Awkward Array layout.

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
    """
    for content in iter_leaf_contents(
        a,
        string_as_leaf=exclude_string,
        bytestring_as_leaf=exclude_bytestring,
    ):
        if isinstance(content, NumpyArray):
            yield content.data
