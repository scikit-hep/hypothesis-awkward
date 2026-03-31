from typing import TYPE_CHECKING

import numpy as np
from hypothesis import assume
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak
from awkward.contents import ByteMaskedArray, Content
from hypothesis_awkward.util.awkward import content_size
from hypothesis_awkward.util.safe import safe_compare as sc

if TYPE_CHECKING:
    from .content import StContent
    from .option import StOption


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
            content = draw(
                st_ak.contents.contents(allow_union_root=False, allow_option_root=False)
            )
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


@st.composite
def byte_masked_array_from_contents(
    draw: st.DrawFn,
    content: 'StContent',
    *,
    max_size: int,
    max_leaf_size: 'int | None',
    max_length: 'int | None',
    st_option: 'StOption | None' = None,
) -> ByteMaskedArray:
    '''Strategy that generates a byte-masked layout within a size limit.

    Called by ``contents()`` during recursive tree generation.

    Parameters
    ----------
    content
        A callable that accepts ``max_size`` and ``max_leaf_size`` and returns
        a strategy for a single content.
    max_size
        Upper bound on ``content_size()`` of the result.
    max_leaf_size
        Upper bound on total leaf elements. ``None`` means no constraint.
    max_length
        Upper bound on ``len(result)``.

    '''
    max_content_size = max((max_size - 1) // 2, 0)
    if max_length is not None:
        max_content_size = min(max_content_size, max_length)
    st_content = content(
        max_size=max_content_size,
        max_leaf_size=max_leaf_size,
        allow_option_root=False,
        allow_union_root=False,
    )
    result = draw(byte_masked_array_contents(st_content))
    assume(content_size(result) <= max_size)
    assume(len(result) <= sc(max_length))
    return result
