from hypothesis import given
from hypothesis import strategies as st

from awkward.contents import (
    EmptyArray,
    ListArray,
    ListOffsetArray,
    NumpyArray,
    RecordArray,
    RegularArray,
)
from hypothesis_awkward import strategies as st_ak
from hypothesis_awkward.util import is_zero_field_record_leaf, iter_leaf_contents


@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """Verify all yielded items are leaf content types."""
    a = data.draw(st_ak.constructors.arrays(), label='array')
    string_as_leaf = data.draw(st.booleans(), label='string_as_leaf')
    bytestring_as_leaf = data.draw(st.booleans(), label='bytestring_as_leaf')
    zero_field_record_as_leaf = data.draw(
        st.booleans(), label='zero_field_record_as_leaf'
    )
    for content in iter_leaf_contents(
        a,
        string_as_leaf=string_as_leaf,
        bytestring_as_leaf=bytestring_as_leaf,
        zero_field_record_as_leaf=zero_field_record_as_leaf,
    ):
        if isinstance(content, RecordArray):
            assert zero_field_record_as_leaf
            assert is_zero_field_record_leaf(content)
        else:
            assert isinstance(
                content,
                (NumpyArray, EmptyArray, ListOffsetArray, ListArray, RegularArray),
            )
        if string_as_leaf:
            assert content.parameter('__array__') != 'char'
        else:
            assert content.parameter('__array__') != 'string'
        if bytestring_as_leaf:
            assert content.parameter('__array__') != 'byte'
        else:
            assert content.parameter('__array__') != 'bytestring'
