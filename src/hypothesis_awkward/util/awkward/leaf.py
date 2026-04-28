from awkward.contents import Content


def is_string_or_bytestring_leaf(
    c: Content,
    string_as_leaf: bool = True,
    bytestring_as_leaf: bool = True,
) -> bool:
    """Return `True` if an [`ak.contents.Content`][] is string or bytestring.

    Parameters
    ----------
    c
        An Awkward [`Content`][ak.contents.Content] node.
    string_as_leaf
        If `True` (default), treat string content as a leaf.
    bytestring_as_leaf
        If `True` (default), treat bytestring content as a leaf.

    Returns
    -------
    bool
        `True` if the content is a string or bytestring leaf.
    """
    return (string_as_leaf and is_string_leaf(c)) or (
        bytestring_as_leaf and is_bytestring_leaf(c)
    )


def is_string_leaf(c: Content) -> bool:
    """Return `True` if an [`ak.contents.Content`][] is string.

    Parameters
    ----------
    c
        An Awkward [`Content`][ak.contents.Content] node.

    Returns
    -------
    bool
        `True` if the content has `__array__` parameter
        `'string'`.
    """
    return c.parameter('__array__') == 'string'


def is_bytestring_leaf(c: Content) -> bool:
    """Return `True` if an [`ak.contents.Content`][] is bytestring.

    Parameters
    ----------
    c
        An Awkward [`Content`][ak.contents.Content] node.

    Returns
    -------
    bool
        `True` if the content has `__array__` parameter
        `'bytestring'`.
    """
    return c.parameter('__array__') == 'bytestring'
