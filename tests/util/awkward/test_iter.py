from hypothesis import given
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak
from awkward.contents import (
    Content,
    EmptyArray,
    ListArray,
    ListOffsetArray,
    NumpyArray,
    RegularArray,
)
from hypothesis_awkward.util import iter_numpy_arrays
from hypothesis_awkward.util.awkward import iter_contents, iter_leaf_contents
from tests.util.awkward.conftest import st_arrays


@given(data=st.data())
def test_iter_numpy_arrays(data: st.DataObject) -> None:
    '''Verify total element count matches len(ak.flatten()).'''
    a = data.draw(st_arrays())
    exclude_string = data.draw(st.booleans(), label='exclude_string')
    exclude_bytestring = data.draw(st.booleans(), label='exclude_bytestring')
    total = sum(
        arr.size
        for arr in iter_numpy_arrays(
            a, exclude_string=exclude_string, exclude_bytestring=exclude_bytestring
        )
    )
    expected = _count_all_leaves(a)
    assert total == expected

    for c in iter_leaf_contents(
        a, string_as_leaf=exclude_string, bytestring_as_leaf=exclude_bytestring
    ):
        if isinstance(c, NumpyArray):
            if exclude_string:
                assert c.parameter('__array__') != 'char'
            if exclude_bytestring:
                assert c.parameter('__array__') != 'byte'


@given(data=st.data())
def test_iter_leaf_contents(data: st.DataObject) -> None:
    '''Verify all yielded items are leaf content types.'''
    a = data.draw(st_arrays())
    string_as_leaf = data.draw(st.booleans(), label='string_as_leaf')
    bytestring_as_leaf = data.draw(st.booleans(), label='bytestring_as_leaf')
    for content in iter_leaf_contents(
        a, string_as_leaf=string_as_leaf, bytestring_as_leaf=bytestring_as_leaf
    ):
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


def _children(c: Content) -> list[Content]:
    '''Return direct Content children of a node.'''
    match c:
        case ak.contents.RecordArray():
            return [c[field] for field in c.fields]
        case ak.contents.UnionArray():
            return list(c.contents)
        case _ if hasattr(c, 'content'):
            return [c.content]
        case _:
            return []


@given(data=st.data())
def test_iter_contents(data: st.DataObject) -> None:
    '''Verify iter_contents yields exactly the full Content tree.'''
    a = data.draw(st_ak.constructors.arrays(), label='array')
    string_as_leaf = data.draw(st.booleans(), label='string_as_leaf')
    bytestring_as_leaf = data.draw(st.booleans(), label='bytestring_as_leaf')
    all_contents = list(
        iter_contents(
            a, string_as_leaf=string_as_leaf, bytestring_as_leaf=bytestring_as_leaf
        )
    )
    id_set = {id(c) for c in all_contents}

    # 1. Type invariant
    assert all(isinstance(c, Content) for c in all_contents)

    # 2. Root inclusion
    assert id(a.layout) in id_set

    # 3. Closure: children of every yielded node are also yielded
    for c in all_contents:
        for child in _children(c):
            assert id(child) in id_set

    # 4. No duplicates
    assert len(id_set) == len(all_contents)

    # 5. String/bytestring parameter invariants
    for c in all_contents:
        if string_as_leaf:
            assert c.parameter('__array__') != 'char'
        if bytestring_as_leaf:
            assert c.parameter('__array__') != 'byte'


def _count_all_leaves(a: ak.Array) -> int:
    '''Count total elements across all leaf fields (handles record arrays).'''
    if a.fields:
        return sum(_count_all_leaves(a[field]) for field in a.fields)
    return len(ak.flatten(a, axis=None))
