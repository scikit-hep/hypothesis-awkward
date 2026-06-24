import numpy as np
from hypothesis import assume
from hypothesis import strategies as st

import awkward as ak
from awkward.contents import Content, IndexedArray
from hypothesis_awkward import strategies as st_ak


@st.composite
def indexed_array_contents(
    draw: st.DrawFn,
    content: st.SearchStrategy[Content] | Content | None = None,
    *,
    min_size: int = 0,
    max_size: int | None = None,
) -> IndexedArray:
    """Strategy for [`ak.contents.IndexedArray`][] instances.

    An [`IndexedArray`][ak.contents.IndexedArray] rearranges the elements of its
    content: each entry of the index selects a content position, so the result can
    reorder, duplicate, or drop elements. It has the same type as its content. The
    index length is drawn independently of the content length. Every entry references
    a valid content position (`0 <= index[i] < len(content)`); duplicates are allowed.
    Unlike [`IndexedOptionArray`][ak.contents.IndexedOptionArray], there are no missing
    entries, so the index also admits the unsigned `uint32` dtype.

    Parameters
    ----------
    content
        Child content. Can be a strategy for [`Content`][ak.contents.Content], a concrete
        [`Content`][ak.contents.Content] instance, or `None` to draw from
        `contents()`.
    min_size
        Lower bound on the index length, i.e., `len(result)`.
    max_size
        Upper bound on the index length, i.e., `len(result)`. If `None`, twice the
        content length is used.

    Returns
    -------
    IndexedArray

    Examples
    --------
    >>> c = indexed_array_contents().example()
    >>> isinstance(c, IndexedArray)
    True

    Limit the index length:

    >>> c = indexed_array_contents(min_size=2, max_size=5).example()
    >>> 2 <= len(c) <= 5
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
    upper = max_size if max_size is not None else max(content_len * 2, min_size)
    if content_len == 0:
        # An empty content can only be referenced by an empty index, so the result
        # has length 0. There is no -1 fallback as in IndexedOptionArray.
        assume(min_size == 0)
        index_list = list[int]()
    else:
        index_list = draw(
            st.lists(
                st.sampled_from(range(content_len)),
                min_size=min_size,
                max_size=upper,
            )
        )
    dtype = draw(st.sampled_from([np.int32, np.uint32, np.int64]))
    index_array = np.array(index_list, dtype=dtype)
    if dtype == np.int32:
        index = ak.index.Index32(index_array)
    elif dtype == np.uint32:
        index = ak.index.IndexU32(index_array)
    else:
        index = ak.index.Index64(index_array)
    return IndexedArray(index, content)
