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

from .iter import get_contents, is_leaf
from .leaf import is_string_or_bytestring_leaf
from .size import content_own_size


@is_leaf.register
def _(c: NumpyArray, /, **_: bool) -> bool:
    return True


@get_contents.register
def _(c: NumpyArray, /, **_: bool) -> tuple[Content, ...]:
    return ()


@content_own_size.register
def _(c: NumpyArray, /) -> int:
    return len(c.data)


@is_leaf.register
def _(c: EmptyArray, /, **_: bool) -> bool:
    return True


@get_contents.register
def _(c: EmptyArray, /, **_: bool) -> tuple[Content, ...]:
    return ()


@content_own_size.register
def _(c: EmptyArray, /) -> int:
    return 0


@is_leaf.register
def _(
    c: RegularArray,
    /,
    *,
    string_as_leaf: bool = True,
    bytestring_as_leaf: bool = True,
) -> bool:
    return is_string_or_bytestring_leaf(c, string_as_leaf, bytestring_as_leaf)


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


@content_own_size.register
def _(c: RegularArray, /) -> int:
    return 1


@is_leaf.register
def _(
    c: ListOffsetArray,
    /,
    *,
    string_as_leaf: bool = True,
    bytestring_as_leaf: bool = True,
) -> bool:
    return is_string_or_bytestring_leaf(c, string_as_leaf, bytestring_as_leaf)


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


@content_own_size.register
def _(c: ListOffsetArray, /) -> int:
    return len(c.offsets.data)


@is_leaf.register
def _(
    c: ListArray,
    /,
    *,
    string_as_leaf: bool = True,
    bytestring_as_leaf: bool = True,
) -> bool:
    return is_string_or_bytestring_leaf(c, string_as_leaf, bytestring_as_leaf)


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


@content_own_size.register
def _(c: ListArray, /) -> int:
    return len(c.starts.data) + len(c.stops.data)


@get_contents.register
def _(c: RecordArray, /, **_: bool) -> tuple[Content, ...]:
    return tuple(c.contents)


@content_own_size.register
def _(c: RecordArray, /) -> int:
    n_fields = 0 if c.is_tuple else len(c.fields)
    return n_fields


@get_contents.register
def _(c: UnionArray, /, **_: bool) -> tuple[Content, ...]:
    return tuple(c.contents)


@content_own_size.register
def _(c: UnionArray, /) -> int:
    return len(c.tags.data) + len(c.index.data)


@get_contents.register
def _(c: BitMaskedArray, /, **_: bool) -> tuple[Content, ...]:
    return (c.content,)


@content_own_size.register
def _(c: BitMaskedArray, /) -> int:
    return 2 + len(c.mask.data)


@get_contents.register
def _(c: ByteMaskedArray, /, **_: bool) -> tuple[Content, ...]:
    return (c.content,)


@content_own_size.register
def _(c: ByteMaskedArray, /) -> int:
    return 1 + len(c.mask.data)


@get_contents.register
def _(c: IndexedOptionArray, /, **_: bool) -> tuple[Content, ...]:
    return (c.content,)


@content_own_size.register
def _(c: IndexedOptionArray, /) -> int:
    return len(c.index.data)


@get_contents.register
def _(c: UnmaskedArray, /, **_: bool) -> tuple[Content, ...]:
    return (c.content,)


@content_own_size.register
def _(c: UnmaskedArray, /) -> int:
    return 0
