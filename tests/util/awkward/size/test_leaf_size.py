from hypothesis import given
from hypothesis import strategies as st

import awkward as ak
from hypothesis_awkward import strategies as st_ak
from hypothesis_awkward.util import (
    iter_contents,
    iter_leaf_contents,
    leaf_size,
)


@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """``leaf_size`` equals the size of the leaf data at every node."""
    content = data.draw(st_ak.contents.contents(), label='content')

    # At a leaf, `iter_leaf_contents(c)` yields `(c,)`, so the assertion
    # reduces to `leaf_size(c) == len(c)`. At a wrapper, both sides traverse
    # to the same leaves and sum their lengths — i.e., the wrapper does not
    # change the leaf total.
    for c in iter_contents(content):
        expected = sum(len(l) for l in iter_leaf_contents(c))
        actual = leaf_size(c)
        assert actual == expected


@given(a=st_ak.constructors.arrays())
def test_accepts_array(a: ak.Array) -> None:
    """``leaf_size`` accepts an ``ak.Array`` as well as a ``Content``."""
    assert leaf_size(a) == leaf_size(a.layout)
