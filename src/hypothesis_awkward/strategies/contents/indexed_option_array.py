from typing import TYPE_CHECKING

import numpy as np
from hypothesis import assume
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak
from awkward.contents import Content, IndexedOptionArray
from hypothesis_awkward.util.awkward import content_size

if TYPE_CHECKING:
    from .content import StContent
    from .option import StOption


@st.composite
def indexed_option_array_contents(
    draw: st.DrawFn,
    content: st.SearchStrategy[Content] | Content | None = None,
    *,
    max_size: int | None = None,
) -> IndexedOptionArray:
    """Strategy for IndexedOptionArray Content wrapping child Content.

    The index length is drawn independently of the content length. Valid
    entries can reference any content position (duplicates allowed).
    Missing entries have ``index[i] = -1``.

    Parameters
    ----------
    content
        Child content. Can be a strategy for Content, a concrete Content instance, or
        ``None`` to draw from ``contents()``.
    max_size
        Upper bound on the index length, i.e., ``len(result)``. When ``None``,
        defaults to twice the content length.

    Examples
    --------
    >>> c = indexed_option_array_contents().example()
    >>> isinstance(c, Content)
    True
    """
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
    content_len = len(content)
    upper = max_size if max_size is not None else content_len * 2
    pool = [-1, *range(content_len)]
    index_list = draw(st.lists(st.sampled_from(pool), max_size=upper))
    dtype = draw(st.sampled_from([np.int32, np.int64]))
    index_array = np.array(index_list, dtype=dtype)
    if dtype == np.int32:
        index = ak.index.Index32(index_array)
    else:
        index = ak.index.Index64(index_array)
    return IndexedOptionArray(index, content)


@st.composite
def indexed_option_array_from_contents(
    draw: st.DrawFn,
    content: 'StContent',
    *,
    max_size: int,
    max_leaf_size: 'int | None' = None,
    max_length: 'int | None' = None,
    st_option: 'StOption | None' = None,
) -> IndexedOptionArray:
    """Strategy that generates an indexed-option layout within a size limit.

    Draws the index length first, then gives the remainder of the budget
    to the child content.

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
    """
    ml = max_length if max_length is not None else max_size
    n = draw(st.integers(min_value=0, max_value=ml))
    max_content_size = max(max_size - n, 0)
    child = draw(
        content(
            max_size=max_content_size,
            max_leaf_size=max_leaf_size,
            allow_option_root=False,
            allow_union_root=False,
        )
    )
    result = draw(indexed_option_array_contents(child, max_size=n))
    assume(content_size(result) <= max_size)
    return result
