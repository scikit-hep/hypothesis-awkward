import numpy as np
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak
from awkward.contents import BitMaskedArray, Content


@st.composite
def bit_masked_array_contents(
    draw: st.DrawFn,
    content: st.SearchStrategy[Content] | Content | None = None,
) -> BitMaskedArray:
    '''Strategy for BitMaskedArray Content wrapping child Content.

    The logical length always equals ``len(content)``. The mask is
    bit-packed into uint8 bytes.

    Parameters
    ----------
    content
        Child content. Can be a strategy for Content, a concrete Content instance, or
        ``None`` to draw from ``contents()``.

    Examples
    --------
    >>> c = bit_masked_array_contents().example()
    >>> isinstance(c, Content)
    True
    '''
    match content:
        case None:
            content = draw(st_ak.contents.contents(allow_union_root=False))
        case st.SearchStrategy():
            content = draw(content)
        case Content():
            pass
    assert isinstance(content, Content)
    length = len(content)
    n_bytes = (length + 7) // 8
    mask = draw(st.lists(st.integers(0, 255), min_size=n_bytes, max_size=n_bytes))
    mask_array = np.array(mask, dtype=np.uint8)
    valid_when = draw(st.booleans())
    lsb_order = draw(st.booleans())
    return BitMaskedArray(
        ak.index.IndexU8(mask_array), content, valid_when, length, lsb_order
    )
