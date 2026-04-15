import functools

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

from .leaf import is_string_or_bytestring_leaf


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
