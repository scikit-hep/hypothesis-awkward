import numpy as np

import awkward as ak
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
from hypothesis_awkward.util import leaf_size


def test_numpy_array() -> None:
    c = NumpyArray(np.array([1, 2, 3]))
    assert leaf_size(c) == 3


def test_empty_array() -> None:
    c = EmptyArray()
    assert leaf_size(c) == 0


def test_regular_array() -> None:
    c = RegularArray(NumpyArray(np.array([1, 2, 3, 4, 5, 6])), size=3)
    assert leaf_size(c) == 6


def test_record_array_named() -> None:
    c = RecordArray(
        [NumpyArray(np.array([1, 2])), NumpyArray(np.array([3, 4]))],
        fields=['x', 'y'],
    )
    assert leaf_size(c) == 4


def test_record_array_tuple() -> None:
    c = RecordArray(
        [NumpyArray(np.array([1, 2])), NumpyArray(np.array([3, 4]))],
        fields=None,
    )
    assert leaf_size(c) == 4


def test_list_offset_array() -> None:
    c = ListOffsetArray(
        ak.index.Index64(np.array([0, 2, 3])),
        NumpyArray(np.array([1, 2, 3])),
    )
    assert leaf_size(c) == 3


def test_list_array() -> None:
    c = ListArray(
        ak.index.Index64(np.array([0, 2])),
        ak.index.Index64(np.array([2, 3])),
        NumpyArray(np.array([1, 2, 3])),
    )
    assert leaf_size(c) == 3


def test_union_array() -> None:
    c = UnionArray(
        tags=ak.index.Index8(np.array([0, 1, 0], dtype=np.int8)),
        index=ak.index.Index64(np.array([0, 0, 1])),
        contents=[
            NumpyArray(np.array([10, 20])),
            NumpyArray(np.array([30])),
        ],
    )
    assert leaf_size(c) == 3


def test_byte_masked_array() -> None:
    c = ByteMaskedArray(
        ak.index.Index8(np.array([1, 0, 1], dtype=np.int8)),
        NumpyArray(np.array([10, 20, 30])),
        valid_when=True,
    )
    assert leaf_size(c) == 3


def test_bit_masked_array() -> None:
    c = BitMaskedArray(
        ak.index.IndexU8(np.array([0b101], dtype=np.uint8)),
        NumpyArray(np.array([10, 20, 30])),
        valid_when=True,
        length=3,
        lsb_order=True,
    )
    assert leaf_size(c) == 3


def test_unmasked_array() -> None:
    c = UnmaskedArray(NumpyArray(np.array([10, 20, 30])))
    assert leaf_size(c) == 3


def test_indexed_option_array() -> None:
    c = IndexedOptionArray(
        ak.index.Index64(np.array([-1, 0, 1])),
        NumpyArray(np.array([10, 20])),
    )
    assert leaf_size(c) == 2


def test_string() -> None:
    a = ak.Array(['hello', 'world'])
    c = ak.to_layout(a)
    assert leaf_size(c) == 2


def test_bytestring() -> None:
    a = ak.Array([b'hello', b'world'])
    c = ak.to_layout(a)
    assert leaf_size(c) == 2


def test_nested() -> None:
    """Test a nested structure: ListOffsetArray wrapping RegularArray wrapping NumpyArray."""
    inner = NumpyArray(np.array([1, 2, 3, 4]))
    regular = RegularArray(inner, size=2)
    outer = ListOffsetArray(
        ak.index.Index64(np.array([0, 1, 2])),
        regular,
    )
    assert leaf_size(outer) == 4


def test_accepts_array() -> None:
    """``leaf_size`` accepts an ``ak.Array`` as well as a ``Content``."""
    a = ak.Array([[1, 2], [3]])
    assert leaf_size(a) == 3
