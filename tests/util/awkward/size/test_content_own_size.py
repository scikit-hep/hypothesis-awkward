import pytest
from hypothesis import given
from hypothesis import strategies as st

from awkward.contents import (
    BitMaskedArray,
    ByteMaskedArray,
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
from hypothesis_awkward.util import content_own_size


@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    content = data.draw(st_ak.contents.contents(), label='content')

    actual = content_own_size(content)

    match content:
        case NumpyArray():
            expected = len(content.data)
        case EmptyArray():
            expected = 0
        case RegularArray():
            expected = 1
        case RecordArray() if not content.is_tuple:
            expected = len(content.fields)
        case RecordArray():
            expected = 0
        case ListOffsetArray():
            expected = len(content.offsets)
        case ListArray():
            expected = len(content.starts) + len(content.stops)
        case UnionArray():
            expected = len(content.tags) + len(content.index)
        case ByteMaskedArray():
            expected = len(content.mask) + 1
        case BitMaskedArray():
            expected = len(content.mask) + 2
        case UnmaskedArray():
            expected = 0
        case IndexedOptionArray():
            expected = len(content.index)
        case _:  # pragma: no cover
            raise TypeError(f'Unexpected content type: {type(content).__name__}')

    assert actual == expected


def test_unregistered_type_raises() -> None:
    """Unknown types dispatch to the base function, which raises ``TypeError``."""

    class _Unknown:
        pass

    with pytest.raises(TypeError, match='Unexpected content type'):
        content_own_size(_Unknown())  # type: ignore[arg-type]


def test_is_extensible() -> None:
    """Register a handler for a new type without modifying ``content_own_size``.

    ``_Marker`` is local to this test, so the registration cannot influence
    any other test's dispatch. ``singledispatch.registry`` is read-only, so no
    cleanup is performed.
    """

    class _Marker:
        pass

    @content_own_size.register
    def _(c: _Marker, /) -> int:
        assert isinstance(c, _Marker)
        return 42

    assert content_own_size(_Marker()) == 42
