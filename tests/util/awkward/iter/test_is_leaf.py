import pytest
from hypothesis import given
from hypothesis import strategies as st

from awkward.contents import (
    BitMaskedArray,
    ByteMaskedArray,
    EmptyArray,
    IndexedArray,
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
    is_bytestring_leaf,
    is_leaf,
    is_string_leaf,
    is_zero_field_record_leaf,
)


@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    content = data.draw(st_ak.contents.contents(), label='content')
    string_as_leaf = data.draw(st.booleans(), label='string_as_leaf')
    bytestring_as_leaf = data.draw(st.booleans(), label='bytestring_as_leaf')
    zero_field_record_as_leaf = data.draw(
        st.booleans(), label='zero_field_record_as_leaf'
    )

    actual = is_leaf(
        content,
        string_as_leaf=string_as_leaf,
        bytestring_as_leaf=bytestring_as_leaf,
        zero_field_record_as_leaf=zero_field_record_as_leaf,
    )

    match content:
        case NumpyArray() | EmptyArray():
            expected = True
        case ListOffsetArray() | ListArray() | RegularArray() if (
            string_as_leaf and is_string_leaf(content)
        ):
            expected = True
        case ListOffsetArray() | ListArray() | RegularArray() if (
            bytestring_as_leaf and is_bytestring_leaf(content)
        ):
            expected = True
        case ListOffsetArray() | ListArray() | RegularArray():
            expected = False
        case RecordArray() if zero_field_record_as_leaf and is_zero_field_record_leaf(
            content
        ):
            expected = True
        case (
            RecordArray()
            | UnionArray()
            | BitMaskedArray()
            | ByteMaskedArray()
            | IndexedArray()
            | IndexedOptionArray()
            | UnmaskedArray()
        ):
            expected = False
        case _:  # pragma: no cover
            raise TypeError(f'Unexpected content type: {type(content).__name__}')

    assert actual == expected


@pytest.mark.parametrize('is_tuple', [True, False])
def test_zero_field_record(is_tuple: bool) -> None:
    """A zero-field record is a leaf by default; the option disables it."""
    # TODO: Delete this test when `contents()` generates the zero-field record
    # leaf; `test_properties` then covers both flag values for it.
    c = RecordArray([], fields=None if is_tuple else [], length=3)
    assert is_leaf(c) is True
    assert is_leaf(c, zero_field_record_as_leaf=False) is False


def test_unregistered_type_returns_false() -> None:
    """Unknown types dispatch to the base function, which returns `False`."""

    class _Unknown:
        pass

    assert is_leaf(_Unknown()) is False  # type: ignore[arg-type]


def test_is_extensible() -> None:
    """Register a handler for a new type without modifying `is_leaf`.

    `_Marker` is local to this test, so the registration cannot influence
    any other test's dispatch. `singledispatch.registry` is read-only, so no
    cleanup is performed.
    """

    class _Marker:
        pass

    @is_leaf.register
    def _(c: _Marker, /, **_: bool) -> bool:
        assert isinstance(c, _Marker)
        return True

    assert is_leaf(_Marker()) is True
