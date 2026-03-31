'''Reproduce ak.from_buffers bug with virtual buffers + BitMaskedArray + RegularArray(size=0).

Fixed on awkward dev after v2.9.0 (likely PR #3889).
'''

import numpy as np
import pytest

import awkward as ak
from awkward.contents import (
    BitMaskedArray,
    ListOffsetArray,
    NumpyArray,
    RegularArray,
)


@pytest.mark.xfail(reason='fails with awkward v2.9.0')
def test_from_buffers_virtual_bitmask_regular_size0() -> None:
    '''
    This test fails with awkward v2.9.0, the latest version as of this writing.

    It will probably be fixed in the next release.
    It passes with the main branch with the head #85c39543.
    It is likely to have been fixed by PR #3889.

    An exception is raised by `ak.from_buffers()`.

    The error message is:

    `AssertionError: RegularArray length must be an integer for an array with concrete
    data, not <class 'awkward._nplikes.shape.UnknownLength'>`

    This error prevent a virtual array from being generated with an empty RegularArray
    with an option type.

    '''
    layout = ListOffsetArray(
        ak.index.Index64(np.array([0, 0])),
        BitMaskedArray(
            ak.index.IndexU8(np.array([], dtype=np.uint8)),
            RegularArray(
                NumpyArray(np.array([], dtype=np.bool_)), size=0, zeros_length=0
            ),
            valid_when=False,
            length=0,
            lsb_order=False,
        ),
    )
    array = ak.Array(layout)
    form, length, buffers = ak.to_buffers(array)
    virtual_buffers = {k: (lambda v=v: v) for k, v in buffers.items()}
    ak.from_buffers(form, length, virtual_buffers)
