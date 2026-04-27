import functools
from collections.abc import Iterator

import numpy as np

import awkward as ak
from awkward.contents import Content, NumpyArray


@functools.singledispatch
def get_contents(
    c: Content,
    /,
    *,
    string_as_leaf: bool = True,
    bytestring_as_leaf: bool = True,
) -> tuple[Content, ...]:
    """Return the direct inner contents of the given content.

    This function receives an instance of a subclass of [`Content`][ak.contents.Content]
    and returns a tuple of its direct inner contents based on the optional arguments.

    This function is a [functools.singledispatch][]. Support for a new
    [`Content`][ak.contents.Content] subclass can be added with
    `get_contents.register()`.

    Parameters
    ----------
    c
        An instance of a subclass of [`Content`][ak.contents.Content].
    string_as_leaf
        Whether to consider a string list as a leaf content. See Examples below.
    bytestring_as_leaf
        Whether to consider a bytestring list as a leaf content. See Examples below.

    Returns
    -------
    tuple[Content, ...]
        The direct inner contents, or an empty tuple if none.

    Examples
    --------
    [`EmptyArray`][ak.contents.EmptyArray] / [`NumpyArray`][ak.contents.NumpyArray]:
    These are leaf contents and have no inner contents.
    The `get_contents()` returns an empty tuple `()`.

    >>> from awkward.contents import EmptyArray, NumpyArray
    >>> c = EmptyArray()
    >>> get_contents(c)
    ()
    >>> c = NumpyArray([1, 2, 3])
    >>> get_contents(c)
    ()

    [`RegularArray`][ak.contents.RegularArray] / [`ListArray`][ak.contents.ListArray] /
    [`ListOffsetArray`][ak.contents.ListOffsetArray]:
    These have one inner content.
    The `get_contents()` returns a tuple with one element of the inner content:

    >>> from awkward.contents import RegularArray, ListArray, ListOffsetArray
    >>> i = ak.from_iter([1, 2, 3, 4, 5, 6], highlevel=False)
    >>> c = RegularArray(i, size=2)
    >>> get_contents(c) == (i,)
    True
    >>> start = ak.index.Index64([0, 3, 3])
    >>> stop = ak.index.Index64([3, 3, 5])
    >>> c = ListArray(start, stop, i)
    >>> get_contents(c) == (i,)
    True
    >>> offsets = ak.index.Index64([0, 3, 3, 5])
    >>> c = ListOffsetArray(offsets, i)
    >>> get_contents(c) == (i,)
    True

    Strings and bytestrings: An array of strings (bytestrings) are a
    [`ListOffsetArray`][ak.contents.ListOffsetArray],
    [`ListArray`][ak.contents.ListArray], or [`RegularArray`][ak.contents.RegularArray]
    with an inner `NumpyArray`. However, by default,`get_contents()` considers them leaf
    contents and returns an empty tuple `()`. With the option `string_as_leaf=False`
    (`bytestring_as_leaf=False`), it returns a tuple with the single content of the
    underlying `NumpyArray`:

    >>> c = ak.from_iter(['abc', 'de'], highlevel=False)
    >>> get_contents(c)
    ()
    >>> get_contents(c, string_as_leaf=False) == (c.content,)
    True
    >>> c = ak.from_iter([b'abc', b'de'], highlevel=False)
    >>> get_contents(c)
    ()
    >>> get_contents(c, bytestring_as_leaf=False) == (c.content,)
    True

    [`RecordArray`][ak.contents.RecordArray] / [`UnionArray`][ak.contents.UnionArray]:
    These have multiple inner contents.
    The `get_contents()` returns a tuple of the inner contents:

    >>> from awkward.contents import RecordArray, UnionArray
    >>> c = ak.zip({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]}, highlevel=False)
    >>> isinstance(c, RecordArray)
    True
    >>> get_contents(c) == tuple(c.contents)
    True
    >>> c = ak.from_iter([0.0, [1, 2], 'three', 4.4, [5]], highlevel=False)
    >>> isinstance(c, UnionArray)
    True
    >>> get_contents(c) == tuple(c.contents)
    True

    [`IndexedOptionArray`][ak.contents.IndexedOptionArray] /
    [`ByteMaskedArray`][ak.contents.ByteMaskedArray] /
    [`BitMaskedArray`][ak.contents.BitMaskedArray] /
    [`UnmaskedArray`][ak.contents.UnmaskedArray]:
    These have one inner content. The
    `get_contents()` returns a tuple with one element of the inner content:

    >>> from awkward.contents import (
    ...     IndexedOptionArray,
    ...     ByteMaskedArray,
    ...     BitMaskedArray,
    ...     UnmaskedArray,
    ... )
    >>> i = ak.from_iter([1, 2, 3, 4, 5, 6], highlevel=False)
    >>> index = ak.index.Index64([0, -1, 2, -1, 4, 5])
    >>> c = IndexedOptionArray(index, i)
    >>> get_contents(c) == (i,)
    True
    >>> mask = ak.index.Index8([1, 0, 1, 0, 1, 1])
    >>> c = ByteMaskedArray(mask, i, valid_when=True)
    >>> get_contents(c) == (i,)
    True
    >>> bitmask = ak.index.IndexU8(np.array([0b10101100], dtype=np.uint8))
    >>> c = BitMaskedArray(bitmask, i, valid_when=True, length=6, lsb_order=False)
    >>> get_contents(c) == (i,)
    True
    >>> c = UnmaskedArray(i)
    >>> get_contents(c) == (i,)
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
        If `True` (default), treat string
        [`ListOffsetArray`][ak.contents.ListOffsetArray]/[`ListArray`][ak.contents.ListArray]/
        [`RegularArray`][ak.contents.RegularArray] nodes as leaves — do not
        descend into the inner [`NumpyArray`][ak.contents.NumpyArray]`(uint8)`.
    bytestring_as_leaf
        If `True` (default), treat bytestring nodes as leaves.

    Yields
    ------
    Content
        Each content node in the layout.
    """
    # TODO: Add an option to skip contents that are unreachable from the
    # outer — e.g., a list's content when no offsets/starts-stops reference
    # it, or an option's content when all mask/index entries exclude it.
    # Planned consumer: `test_draw_max_length_not_recursed` in
    # tests/strategies/contents/test_content.py, to assert that max_length
    # does not constrain genuinely reachable inner contents.
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
) -> Iterator[Content]:
    """Iterate over all leaf contents in an Awkward Array layout.

    Parameters
    ----------
    a
        An Awkward Array or Content.
    string_as_leaf
        If `True` (default), treat string
        [`ListOffsetArray`][ak.contents.ListOffsetArray]/[`ListArray`][ak.contents.ListArray]/
        [`RegularArray`][ak.contents.RegularArray] nodes as leaves.
    bytestring_as_leaf
        If `True` (default), treat bytestring nodes as leaves.

    Yields
    ------
    Content
        Each leaf content in the layout.
    """
    for content in iter_contents(
        a, string_as_leaf=string_as_leaf, bytestring_as_leaf=bytestring_as_leaf
    ):
        if is_leaf(
            content,
            string_as_leaf=string_as_leaf,
            bytestring_as_leaf=bytestring_as_leaf,
        ):
            yield content


@functools.singledispatch
def is_leaf(
    c: Content,
    /,
    *,
    string_as_leaf: bool = True,
    bytestring_as_leaf: bool = True,
) -> bool:
    """Return ``True`` if an [`ak.contents.Content`][] is a leaf.

    [`NumpyArray`][ak.contents.NumpyArray] and [`EmptyArray`][ak.contents.EmptyArray] are always leaves. String and
    bytestring list nodes are leaves only when the respective flag is
    set. Wrappers ([`RecordArray`][ak.contents.RecordArray], [`UnionArray`][ak.contents.UnionArray], option/masked types,
    non-string list types) are never leaves. Unknown types fall back to
    ``False``.

    Dispatch is performed with [functools.singledispatch][] so support
    for a new [`Content`][ak.contents.Content] subclass can be added by calling
    ``is_leaf.register`` without modifying this function.

    Parameters
    ----------
    c
        An Awkward [`Content`][ak.contents.Content] node.
    string_as_leaf
        If ``True`` (default), treat string [`ListOffsetArray`][ak.contents.ListOffsetArray]/
        [`ListArray`][ak.contents.ListArray]/[`RegularArray`][ak.contents.RegularArray] nodes as leaves.
    bytestring_as_leaf
        Same as ``string_as_leaf`` for bytestring nodes.

    Returns
    -------
    bool
        ``True`` if ``c`` is a leaf under the given flags, else
        ``False``.

    Examples
    --------
    >>> import numpy as np
    >>> from awkward.contents import NumpyArray, RegularArray
    >>> is_leaf(NumpyArray(np.array([1, 2, 3])))
    True

    A non-string [`RegularArray`][ak.contents.RegularArray] is not a leaf:

    >>> c = RegularArray(NumpyArray(np.array([1, 2, 3, 4])), size=2)
    >>> is_leaf(c)
    False
    """
    return False


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
