import pytest
from hypothesis import given
from hypothesis import strategies as st

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
from hypothesis_awkward import strategies as st_ak
from hypothesis_awkward.util import (
    get_contents,
    is_bytestring_leaf,
    is_string_leaf,
)


@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    content = data.draw(st_ak.contents.contents(), label='content')
    string_as_leaf = data.draw(st.booleans(), label='string_as_leaf')
    bytestring_as_leaf = data.draw(st.booleans(), label='bytestring_as_leaf')

    actual = get_contents(
        content,
        string_as_leaf=string_as_leaf,
        bytestring_as_leaf=bytestring_as_leaf,
    )

    match content:
        case NumpyArray() | EmptyArray():
            expected: tuple[Content, ...] = ()
        case ListOffsetArray() | ListArray() | RegularArray() if (
            string_as_leaf and is_string_leaf(content)
        ):
            expected = ()
        case ListOffsetArray() | ListArray() | RegularArray() if (
            bytestring_as_leaf and is_bytestring_leaf(content)
        ):
            expected = ()
        case ListOffsetArray() | ListArray() | RegularArray():
            expected = (content.content,)
        case RecordArray() | UnionArray():
            expected = tuple(content.contents)
        case (
            BitMaskedArray()
            | ByteMaskedArray()
            | IndexedOptionArray()
            | UnmaskedArray()
        ):
            expected = (content.content,)
        case _:  # pragma: no cover
            raise TypeError(f'Unexpected content type: {type(content).__name__}')

    assert actual == expected


def test_unregistered_type_raises() -> None:
    """Unknown types dispatch to the base function, which raises ``TypeError``."""

    class _Unknown:
        pass

    with pytest.raises(TypeError, match='Unexpected content type'):
        get_contents(_Unknown())  # type: ignore[arg-type]


def test_is_extensible() -> None:
    """Register a handler for a new type without modifying ``get_contents``.

    ``_Marker`` is local to this test, so the registration cannot influence
    any other test's dispatch. ``singledispatch.registry`` is read-only, so no
    cleanup is performed.
    """

    class _Marker:
        pass

    @get_contents.register
    def _(c: _Marker, /, **_: bool) -> tuple[Content, ...]:
        assert isinstance(c, _Marker)
        return ()

    assert get_contents(_Marker()) == ()
