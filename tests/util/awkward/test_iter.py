from hypothesis import given
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak
from awkward.contents import Content
from hypothesis_awkward.util import iter_numpy_arrays
from hypothesis_awkward.util.awkward import iter_contents, iter_leaf_contents
from tests.util.awkward.conftest import st_arrays


@given(data=st.data())
def test_iter_numpy_arrays(data: st.DataObject) -> None:
    '''Verify total element count matches len(ak.flatten()).'''
    a = data.draw(st_arrays())
    total = sum(arr.size for arr in iter_numpy_arrays(a))
    expected = _count_all_leaves(a)
    assert total == expected


@given(data=st.data())
def test_iter_leaf_contents(data: st.DataObject) -> None:
    '''Verify all yielded items are EmptyArray or NumpyArray.'''
    a = data.draw(st_arrays())
    for content in iter_leaf_contents(a):
        assert isinstance(content, (ak.contents.EmptyArray, ak.contents.NumpyArray))


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
    all_contents = list(iter_contents(a))
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


def _count_all_leaves(a: ak.Array) -> int:
    '''Count total elements across all leaf fields (handles record arrays).'''
    if a.fields:
        return sum(_count_all_leaves(a[field]) for field in a.fields)
    return len(ak.flatten(a, axis=None))
