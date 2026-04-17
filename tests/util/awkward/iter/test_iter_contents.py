from hypothesis import given
from hypothesis import strategies as st

import awkward as ak
from awkward.contents import Content
from hypothesis_awkward import strategies as st_ak
from hypothesis_awkward.util import iter_contents


@given(data=st.data())
def test_iter_contents(data: st.DataObject) -> None:
    """Verify iter_contents yields exactly the full Content tree."""
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
    #    (string/bytestring leaves don't descend, so skip their children)
    for c in all_contents:
        if string_as_leaf and c.parameter('__array__') == 'string':
            continue
        if bytestring_as_leaf and c.parameter('__array__') == 'bytestring':
            continue
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

    # 6. DFS preorder, natural left-to-right child order
    expected = _dfs_pre_order(
        a.layout,
        string_as_leaf=string_as_leaf,
        bytestring_as_leaf=bytestring_as_leaf,
    )
    assert [id(c) for c in all_contents] == [id(c) for c in expected]


def _children(c: Content) -> list[Content]:
    """Return direct Content children of a node."""
    match c:
        case ak.contents.RecordArray():
            return list(c.contents)
        case ak.contents.UnionArray():
            return list(c.contents)
        case _ if hasattr(c, 'content'):
            return [c.content]
        case _:
            return []


def _dfs_pre_order(
    c: Content, *, string_as_leaf: bool, bytestring_as_leaf: bool
) -> list[Content]:
    """Reference DFS pre-order traversal honoring string/bytestring leaf flags."""
    out = [c]
    if string_as_leaf and c.parameter('__array__') == 'string':
        return out
    if bytestring_as_leaf and c.parameter('__array__') == 'bytestring':
        return out
    for child in _children(c):
        out.extend(
            _dfs_pre_order(
                child,
                string_as_leaf=string_as_leaf,
                bytestring_as_leaf=bytestring_as_leaf,
            )
        )
    return out
