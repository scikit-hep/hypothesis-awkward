import functools
from collections.abc import Iterator
from typing import Union

import numpy as np

import awkward as ak
from awkward.contents import (
    BitMaskedArray,
    ByteMaskedArray,
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


def any_nan_nat_in_awkward_array(a: ak.Array | Content, /) -> bool:
    """`True` if Awkward Array contains any `NaN` or `NaT` values, else `False`.

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
    """
    return any_nan_in_awkward_array(a) or any_nat_in_awkward_array(a)


def any_nan_in_awkward_array(a: ak.Array | Content, /) -> bool:
    """`True` if Awkward Array contains any `NaN` values, else `False`.

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
    """
    for arr in iter_numpy_arrays(a):
        if arr.dtype.kind in {'f', 'c'} and np.any(np.isnan(arr)):
            return True
    return False


def any_nat_in_awkward_array(a: ak.Array | Content, /) -> bool:
    """`True` if Awkward Array contains any `NaT` values, else `False`.

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
    """
    for arr in iter_numpy_arrays(a):
        if arr.dtype.kind in {'m', 'M'} and np.any(np.isnat(arr)):
            return True
    return False


def is_string_or_bytestring_leaf(
    c: Content,
    string_as_leaf: bool = True,
    bytestring_as_leaf: bool = True,
) -> bool:
    """Check whether an Awkward Content node is a string or bytestring leaf.

    Parameters
    ----------
    c
        An Awkward Content node.
    string_as_leaf
        If ``True`` (default), treat string content as a leaf.
    bytestring_as_leaf
        If ``True`` (default), treat bytestring content as a leaf.

    Returns
    -------
    bool
        ``True`` if the content is a string or bytestring leaf.
    """
    return (string_as_leaf and is_string_leaf(c)) or (
        bytestring_as_leaf and is_bytestring_leaf(c)
    )


def is_string_leaf(c: Content) -> bool:
    """Check whether an Awkward Content node is a string leaf.

    Parameters
    ----------
    c
        An Awkward Content node.

    Returns
    -------
    bool
        ``True`` if the content has ``__array__`` parameter
        ``'string'``.
    """
    return c.parameter('__array__') == 'string'


def is_bytestring_leaf(c: Content) -> bool:
    """Check whether an Awkward Content node is a bytestring leaf.

    Parameters
    ----------
    c
        An Awkward Content node.

    Returns
    -------
    bool
        ``True`` if the content has ``__array__`` parameter
        ``'bytestring'``.
    """
    return c.parameter('__array__') == 'bytestring'


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
    >>> c = NumpyArray(np.array([1, 2, 3]))
    >>> get_contents(c)
    ()

    >>> c = RegularArray(NumpyArray(np.array([1, 2, 3, 4])), size=2)
    >>> subs = get_contents(c)
    >>> len(subs) == 1 and subs[0] is c.content
    True
    """
    raise TypeError(f'Unexpected content type: {type(c)}')  # pragma: no cover


@get_contents.register
def _(c: NumpyArray, /, **_: bool) -> tuple[Content, ...]:
    return ()


@get_contents.register
def _(c: EmptyArray, /, **_: bool) -> tuple[Content, ...]:
    return ()


@get_contents.register
def _(
    c: RegularArray,
    /,
    *,
    string_as_leaf: bool = True,
    bytestring_as_leaf: bool = True,
) -> tuple[Content, ...]:
    if is_string_or_bytestring_leaf(c, string_as_leaf, bytestring_as_leaf):
        return ()
    return (c.content,)


@get_contents.register
def _(
    c: ListOffsetArray,
    /,
    *,
    string_as_leaf: bool = True,
    bytestring_as_leaf: bool = True,
) -> tuple[Content, ...]:
    if is_string_or_bytestring_leaf(c, string_as_leaf, bytestring_as_leaf):
        return ()
    return (c.content,)


@get_contents.register
def _(
    c: ListArray,
    /,
    *,
    string_as_leaf: bool = True,
    bytestring_as_leaf: bool = True,
) -> tuple[Content, ...]:
    if is_string_or_bytestring_leaf(c, string_as_leaf, bytestring_as_leaf):
        return ()
    return (c.content,)


@get_contents.register
def _(c: RecordArray, /, **_: bool) -> tuple[Content, ...]:
    return tuple(c.contents)


@get_contents.register
def _(c: UnionArray, /, **_: bool) -> tuple[Content, ...]:
    return tuple(c.contents)


@get_contents.register
def _(c: BitMaskedArray, /, **_: bool) -> tuple[Content, ...]:
    return (c.content,)


@get_contents.register
def _(c: ByteMaskedArray, /, **_: bool) -> tuple[Content, ...]:
    return (c.content,)


@get_contents.register
def _(c: IndexedOptionArray, /, **_: bool) -> tuple[Content, ...]:
    return (c.content,)


@get_contents.register
def _(c: UnmaskedArray, /, **_: bool) -> tuple[Content, ...]:
    return (c.content,)


def leaf_size(a: ak.Array | Content, /) -> int:
    """Count total leaf elements in an Awkward Array layout.

    Each ``NumpyArray`` element counts as one. Each string and bytestring
    (not character or byte) counts as one. ``EmptyArray`` counts as zero.

    Parameters
    ----------
    a
        An Awkward Array or Content.

    Returns
    -------
    int
        Total number of leaf elements.

    Examples
    --------
    >>> a = ak.Array([1, 2, 3])
    >>> leaf_size(a)
    3

    >>> a = ak.Array([[1, 2], [3]])
    >>> leaf_size(a)
    3

    >>> a = ak.Array(['hello', 'world'])
    >>> leaf_size(a)
    2
    """
    return sum(len(leaf) for leaf in iter_leaf_contents(a))


def content_size(a: ak.Array | Content, /) -> int:
    """Count total scalars stored in an Awkward Array layout.

    Counts data elements, offset/index buffer elements, and metadata values
    (``RegularArray.size``, ``RecordArray`` field names).

    Parameters
    ----------
    a
        An Awkward Array or Content.

    Returns
    -------
    int
        Total number of scalars stored in the content tree.

    Examples
    --------
    A flat array has content_size equal to its length:

    >>> a = ak.Array([1, 2, 3])
    >>> content_size(a)
    3

    A variable-length list array counts offsets (n+1) plus child data:

    >>> a = ak.Array([[1, 2], [3]])
    >>> content_size(a)  # 3 offsets + 3 data = 6
    6

    A string array counts offsets (n+1) plus UTF-8 bytes:

    >>> a = ak.Array(['hello', 'world'])
    >>> content_size(a)  # 3 offsets + 10 bytes = 13
    13
    """
    match a:
        case ak.Array():
            return content_size(a.layout)
        case Content():
            return content_own_size(a) + _inner_contents_size(a)
        case _:
            raise TypeError(f'Unexpected content type: {type(a)}')  # pragma: no cover


def _inner_contents_size(content: Content, /) -> int:
    return sum(
        content_size(c)
        for c in get_contents(content, string_as_leaf=False, bytestring_as_leaf=False)
    )


@functools.singledispatch
def content_own_size(c: Content, /) -> int:
    """Count the scalars owned directly by a ``Content`` node.

    Counts the node's own buffer elements (offsets, starts/stops, mask, tags,
    index, numeric data) and metadata values (``RegularArray.size``,
    ``RecordArray`` field names, ``BitMaskedArray``/``ByteMaskedArray`` flags).
    Does **not** recurse into sub-contents — that is handled by
    [content_size][hypothesis_awkward.util.content_size] via
    [get_contents][hypothesis_awkward.util.get_contents].

    Dispatch is performed with [functools.singledispatch][] so support for a
    new ``Content`` subclass can be added by calling
    ``content_own_size.register`` without modifying this function.

    Parameters
    ----------
    c
        An Awkward ``Content`` node.

    Returns
    -------
    int
        Number of scalars stored directly on ``c``, excluding sub-contents.

    Examples
    --------
    >>> c = NumpyArray(np.array([1, 2, 3]))
    >>> content_own_size(c)
    3

    The size metadata of a ``RegularArray`` counts as one; the inner
    ``NumpyArray`` data is *not* included here:

    >>> c = RegularArray(NumpyArray(np.array([1, 2, 3, 4])), size=2)
    >>> content_own_size(c)
    1
    """
    raise TypeError(f'Unexpected content type: {type(c)}')  # pragma: no cover


@content_own_size.register
def _(c: NumpyArray, /) -> int:
    return len(c.data)


@content_own_size.register
def _(c: EmptyArray, /) -> int:
    return 0


@content_own_size.register
def _(c: RegularArray, /) -> int:
    return 1


@content_own_size.register
def _(c: RecordArray, /) -> int:
    n_fields = 0 if c.is_tuple else len(c.fields)
    return n_fields


@content_own_size.register
def _(c: ListOffsetArray, /) -> int:
    return len(c.offsets.data)


@content_own_size.register
def _(c: ListArray, /) -> int:
    return len(c.starts.data) + len(c.stops.data)


@content_own_size.register
def _(c: UnionArray, /) -> int:
    return len(c.tags.data) + len(c.index.data)


@content_own_size.register
def _(c: BitMaskedArray, /) -> int:
    return 2 + len(c.mask.data)


@content_own_size.register
def _(c: ByteMaskedArray, /) -> int:
    return 1 + len(c.mask.data)


@content_own_size.register
def _(c: IndexedOptionArray, /) -> int:
    return len(c.index.data)


@content_own_size.register
def _(c: UnmaskedArray, /) -> int:
    return 0
