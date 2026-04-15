import functools
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

from .leaf import is_string_or_bytestring_leaf

LeafContent = Union[NumpyArray, EmptyArray, ListOffsetArray, ListArray, RegularArray]


@functools.singledispatch
def get_contents(
    c: Content,
    /,
    *,
    string_as_leaf: bool = True,
    bytestring_as_leaf: bool = True,
) -> tuple[Content, ...]:
    """Return the direct sub-contents of an Awkward ``Content`` node.

    Dispatch is performed with [functools.singledispatch][] so support for a
    new ``Content`` subclass can be added by calling ``get_contents.register``
    without modifying this function.

    Parameters
    ----------
    c
        An Awkward ``Content`` node. ``ak.Array`` is not accepted — unwrap
        with ``a.layout`` first.
    string_as_leaf
        If ``True`` (default), treat string ``ListOffsetArray``/``ListArray``/
        ``RegularArray`` nodes as leaves — return ``()`` rather than
        ``(c.content,)``.
    bytestring_as_leaf
        Same as ``string_as_leaf`` for bytestring nodes.

    Returns
    -------
    tuple of Content
        The direct sub-contents, in natural order (field order for
        ``RecordArray``, member order for ``UnionArray``). Empty for
        ``NumpyArray``, ``EmptyArray``, and list types configured as leaves.

    Examples
    --------
    >>> import numpy as np
    >>> from awkward.contents import NumpyArray, RegularArray
    >>> c = NumpyArray(np.array([1, 2, 3]))
    >>> get_contents(c)
    ()

    >>> c = RegularArray(NumpyArray(np.array([1, 2, 3, 4])), size=2)
    >>> subs = get_contents(c)
    >>> len(subs) == 1 and subs[0] is c.content
    True
    """
    raise TypeError(f'Unexpected content type: {type(c)}')  # pragma: no cover


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
        stack.extend(reversed(contents))


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
