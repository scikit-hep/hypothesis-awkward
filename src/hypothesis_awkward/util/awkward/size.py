import functools

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

from .contents import get_contents
from .iter import iter_leaf_contents


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
    >>> import numpy as np
    >>> from awkward.contents import NumpyArray, RegularArray
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
