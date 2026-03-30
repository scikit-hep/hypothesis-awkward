import numpy as np
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak
from awkward.contents import Content, IndexedOptionArray


@st.composite
def indexed_option_array_contents(
    draw: st.DrawFn,
    content: st.SearchStrategy[Content] | Content | None = None,
    *,
    max_size: int | None = None,
) -> IndexedOptionArray:
    '''Strategy for IndexedOptionArray Content wrapping child Content.

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
    '''
    match content:
        case None:
            content = draw(st_ak.contents.contents(allow_union_root=False))
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
