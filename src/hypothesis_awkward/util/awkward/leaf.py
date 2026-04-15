from awkward.contents import Content


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
