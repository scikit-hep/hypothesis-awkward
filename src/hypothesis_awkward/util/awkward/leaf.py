from awkward.contents import Content, RecordArray


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


def is_zero_field_record_leaf(c: Content) -> bool:
    """Return `True` if an [`ak.contents.Content`][] is a zero-field record.

    A [`RecordArray`][ak.contents.RecordArray] with no fields is the third, corner-case
    leaf type of a layout tree: it has no children and stores no data.

    Parameters
    ----------
    c
        An Awkward [`Content`][ak.contents.Content] node.

    Returns
    -------
    bool
        `True` if the content is a [`RecordArray`][ak.contents.RecordArray] with no
        fields.
    """
    return isinstance(c, RecordArray) and not c.contents
