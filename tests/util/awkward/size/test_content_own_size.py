import numpy as np
import pytest

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
from hypothesis_awkward.util import content_own_size


def test_numpy_array() -> None:
    """``NumpyArray`` owns its data buffer."""
    c = NumpyArray(np.array([1, 2, 3]))
    assert content_own_size(c) == 3


def test_empty_array() -> None:
    """``EmptyArray`` owns nothing."""
    c = EmptyArray()
    assert content_own_size(c) == 0


def test_regular_array() -> None:
    """``RegularArray`` owns only its ``size`` metadata — not the child."""
    c = RegularArray(NumpyArray(np.array([1, 2, 3, 4, 5, 6])), size=3)
    assert content_own_size(c) == 1


def test_record_array_named() -> None:
    """``RecordArray`` with named fields owns one scalar per field name."""
    c = RecordArray(
        [NumpyArray(np.array([1, 2])), NumpyArray(np.array([3, 4]))],
        fields=['x', 'y'],
    )
    assert content_own_size(c) == 2


def test_record_array_tuple() -> None:
    """``RecordArray`` as a tuple (no field names) owns nothing itself."""
    c = RecordArray(
        [NumpyArray(np.array([1, 2])), NumpyArray(np.array([3, 4]))],
        fields=None,
    )
    assert content_own_size(c) == 0


def test_list_offset_array() -> None:
    """``ListOffsetArray`` owns its offsets buffer (n+1 entries)."""
    c = ListOffsetArray(
        ak.index.Index64(np.array([0, 2, 3])),
        NumpyArray(np.array([1, 2, 3])),
    )
    assert content_own_size(c) == 3


def test_list_array() -> None:
    """``ListArray`` owns its starts and stops buffers."""
    c = ListArray(
        ak.index.Index64(np.array([0, 2])),
        ak.index.Index64(np.array([2, 3])),
        NumpyArray(np.array([1, 2, 3])),
    )
    assert content_own_size(c) == 4


def test_union_array() -> None:
    """``UnionArray`` owns its tags and index buffers."""
    c = UnionArray(
        tags=ak.index.Index8(np.array([0, 1, 0], dtype=np.int8)),
        index=ak.index.Index64(np.array([0, 0, 1])),
        contents=[
            NumpyArray(np.array([10, 20])),
            NumpyArray(np.array([30])),
        ],
    )
    assert content_own_size(c) == 6


def test_byte_masked_array() -> None:
    """``ByteMaskedArray`` owns its mask plus the ``valid_when`` flag."""
    c = ByteMaskedArray(
        ak.index.Index8(np.array([1, 0, 1], dtype=np.int8)),
        NumpyArray(np.array([10, 20, 30])),
        valid_when=True,
    )
    assert content_own_size(c) == 4


def test_bit_masked_array() -> None:
    """``BitMaskedArray`` owns its mask bytes plus ``valid_when`` and ``lsb_order``."""
    c = BitMaskedArray(
        ak.index.IndexU8(np.array([0b101], dtype=np.uint8)),
        NumpyArray(np.array([10, 20, 30])),
        valid_when=True,
        length=3,
        lsb_order=True,
    )
    assert content_own_size(c) == 3


def test_unmasked_array() -> None:
    """``UnmaskedArray`` carries no buffers of its own."""
    c = UnmaskedArray(NumpyArray(np.array([10, 20, 30])))
    assert content_own_size(c) == 0


def test_indexed_option_array() -> None:
    """``IndexedOptionArray`` owns its index buffer."""
    c = IndexedOptionArray(
        ak.index.Index64(np.array([-1, 0, 1])),
        NumpyArray(np.array([10, 20])),
    )
    assert content_own_size(c) == 3


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
