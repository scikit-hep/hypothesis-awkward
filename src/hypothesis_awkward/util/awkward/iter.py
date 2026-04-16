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
    """Return the direct sub-contents of an [`ak.contents.Content`][] node.

    Dispatch is performed with [functools.singledispatch][] so support for a
    new [`Content`][ak.contents.Content] subclass can be added by calling ``get_contents.register``
    without modifying this function.

    Parameters
    ----------
    c
        An Awkward [`Content`][ak.contents.Content] node. [`ak.Array`][ak.Array] is not accepted — unwrap
        with ``a.layout`` first.
    string_as_leaf
        If ``True`` (default), treat string [`ListOffsetArray`][ak.contents.ListOffsetArray]/[`ListArray`][ak.contents.ListArray]/
        [`RegularArray`][ak.contents.RegularArray] nodes as leaves — return ``()`` rather than
        ``(c.content,)``.
    bytestring_as_leaf
        Same as ``string_as_leaf`` for bytestring nodes.

    Returns
    -------
    tuple[Content, ...]
        The direct sub-contents, in natural order (field order for
        [`RecordArray`][ak.contents.RecordArray], member order for [`UnionArray`][ak.contents.UnionArray]). Empty for
        [`NumpyArray`][ak.contents.NumpyArray], [`EmptyArray`][ak.contents.EmptyArray], and list types configured as leaves.

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
