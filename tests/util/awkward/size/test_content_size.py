from hypothesis import given
from hypothesis import strategies as st

import awkward as ak
from hypothesis_awkward import strategies as st_ak
from hypothesis_awkward.util import (
    content_own_size,
    content_size,
    get_contents,
    iter_contents,
)


@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """``content_size`` equals own size plus children's sizes at every node."""
    content = data.draw(st_ak.contents.contents(), label='content')

    # This pins the aggregation rule of `content_size` against
    # `content_own_size` as a primitive. `content_own_size` is independently
    # tested per type in `test_content_own_size.py`; together the two
    # transitively pin `content_size` to the correct per-type formulas.
    for c in iter_contents(content, string_as_leaf=False, bytestring_as_leaf=False):
        inner = get_contents(c, string_as_leaf=False, bytestring_as_leaf=False)
        inner_size = sum(content_size(i) for i in inner)
        expected = content_own_size(c) + inner_size
        actual = content_size(c)
        assert actual == expected


@given(a=st_ak.constructors.arrays())
def test_accepts_array(a: ak.Array) -> None:
    """``content_size`` accepts an ``ak.Array`` as well as a ``Content``."""
    assert content_size(a) == content_size(a.layout)
