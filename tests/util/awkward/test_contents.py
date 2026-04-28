import sys
from typing import cast

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
from hypothesis_awkward.util import (
    get_contents,
    is_bytestring_leaf,
    is_string_leaf,
)

if sys.version_info >= (3, 11):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict


class GetContentsKwargs(TypedDict, total=False):
    """Kwargs of `get_contents()`."""

    string_as_leaf: bool
    bytestring_as_leaf: bool


@st.composite
def get_contents_kwargs(draw: st.DrawFn) -> GetContentsKwargs:
    """Strategy for kwargs of `get_contents()`."""
    kwargs = draw(
        st.fixed_dictionaries(
            {},
            optional={
                'string_as_leaf': st.booleans(),
                'bytestring_as_leaf': st.booleans(),
            },
        )
    )
    return cast(GetContentsKwargs, kwargs)


@given(data=st.data())
def test_get_contents(data: st.DataObject) -> None:

    layout = data.draw(st_ak.contents.contents(), label='content')

    # Draw options
    kwargs = data.draw(get_contents_kwargs(), label='kwargs')

    # Call the test subject
    result = get_contents(layout, **kwargs)

    string_as_leaf = kwargs.get('string_as_leaf', True)
    bytestring_as_leaf = kwargs.get('bytestring_as_leaf', True)

    match layout:
        case NumpyArray() | EmptyArray():
            assert result == ()
        case RecordArray() | UnionArray():
            assert len(result) == len(layout.contents)
            for r, e in zip(result, layout.contents):
                assert r is e
        case ListArray() | ListOffsetArray() | RegularArray() if (
            string_as_leaf and is_string_leaf(layout)
        ):
            assert result == ()
        case ListArray() | ListOffsetArray() | RegularArray() if (
            bytestring_as_leaf and is_bytestring_leaf(layout)
        ):
            assert result == ()
        case (
            BitMaskedArray()
            | ByteMaskedArray()
            | IndexedOptionArray()
            | ListArray()
            | ListOffsetArray()
            | RegularArray()
            | UnmaskedArray()
        ):
            assert len(result) == 1
            assert result[0] is layout.content
        case _:  # pragma: no cover
            raise AssertionError(f'Unexpected content type: {type(layout)}')


def test_get_contents_is_extensible() -> None:
    """Register a handler for a new type without modifying `get_contents`.

    `_Marker` is local to this test, so the registration cannot influence
    any other test's dispatch. `singledispatch.registry` is read-only, so
    no cleanup is performed.
    """

    class _Marker:
        pass

    @get_contents.register
    def _(c: _Marker, /, **_: bool) -> tuple:
        return ()

    assert get_contents(_Marker()) == ()
