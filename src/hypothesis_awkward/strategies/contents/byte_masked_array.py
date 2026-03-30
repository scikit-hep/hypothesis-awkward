import numpy as np
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak
from awkward.contents import ByteMaskedArray, Content


@st.composite
def byte_masked_array_contents(
    draw: st.DrawFn,
    content: st.SearchStrategy[Content] | Content | None = None,
) -> ByteMaskedArray:
    '''Strategy for ByteMaskedArray Content wrapping child Content.

    The mask length always equals ``len(content)``.

    Parameters
    ----------
    content
        Child content. Can be a strategy for Content, a concrete Content instance, or
        ``None`` to draw from ``contents()``.

    Examples
    --------
    >>> c = byte_masked_array_contents().example()
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
    n = len(content)
    mask = draw(st.lists(st.booleans(), min_size=n, max_size=n))
    mask_array = np.array(mask, dtype=np.int8)
    valid_when = draw(st.booleans())
    return ByteMaskedArray(ak.index.Index8(mask_array), content, valid_when)
