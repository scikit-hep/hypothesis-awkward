import pytest
from hypothesis import given
from hypothesis import strategies as st

import awkward as ak
from awkward.contents import RecordArray
from hypothesis_awkward import strategies as st_ak
from hypothesis_awkward.util import (
    iter_contents,
    iter_leaf_contents,
    leaf_size,
)


@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """`leaf_size` equals the size of the leaf data at every node."""
    content = data.draw(st_ak.contents.contents(), label='content')
    zero_field_record_as_leaf = data.draw(
        st.booleans(), label='zero_field_record_as_leaf'
    )

    # At a leaf, `iter_leaf_contents(c)` yields `(c,)`, so the assertion
    # reduces to `leaf_size(c) == len(c)`. At a wrapper, both sides traverse
    # to the same leaves and sum their lengths — i.e., the wrapper does not
    # change the leaf total.
    for c in iter_contents(content):
        expected = sum(
            len(l)
            for l in iter_leaf_contents(
                c, zero_field_record_as_leaf=zero_field_record_as_leaf
            )
        )
        actual = leaf_size(c, zero_field_record_as_leaf=zero_field_record_as_leaf)
        assert actual == expected


@pytest.mark.parametrize('is_tuple', [True, False])
def test_zero_field_record(is_tuple: bool) -> None:
    """A zero-field record counts its length unless disabled."""
    # TODO: Delete this test when `contents()` generates the zero-field record
    # leaf; `test_properties` then covers both flag values for it.
    c = RecordArray([], fields=None if is_tuple else [], length=3)
    assert leaf_size(c) == 3
    assert leaf_size(c, zero_field_record_as_leaf=False) == 0


@given(a=st_ak.constructors.arrays())
def test_accepts_array(a: ak.Array) -> None:
    """`leaf_size` accepts an `ak.Array` as well as a `Content`."""
    assert leaf_size(a) == leaf_size(a.layout)
