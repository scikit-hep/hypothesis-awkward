from typing import TYPE_CHECKING

import numpy as np
from hypothesis import assume
from hypothesis import strategies as st

import awkward as ak
from awkward.contents import Content, IndexedOptionArray
from hypothesis_awkward import strategies as st_ak
from hypothesis_awkward.util import content_size

if TYPE_CHECKING:
    from .content import StContent
    from .option import StOption


@st.composite
def indexed_option_array_contents(
    draw: st.DrawFn,
    content: st.SearchStrategy[Content] | Content | None = None,
    *,
    min_size: int = 0,
    max_size: int | None = None,
) -> IndexedOptionArray:
    """Strategy for [`ak.contents.IndexedOptionArray`][] instances.

    The index length is drawn independently of the content length. Valid entries can
    reference any content position (duplicates allowed). Missing entries have `index[i]
    = -1`.

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
    IndexedOptionArray

    Examples
    --------
    >>> c = indexed_option_array_contents().example()
    >>> isinstance(c, IndexedOptionArray)
    True

    Limit the index length:

    >>> c = indexed_option_array_contents(min_size=2, max_size=5).example()
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
    pool = [-1, *range(content_len)]
    index_list = draw(
        st.lists(st.sampled_from(pool), min_size=min_size, max_size=upper)
    )
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
    max_leaf_size: int | None = None,
    min_length: int = 0,
    max_length: int | None = None,
    st_option: 'StOption | None' = None,
) -> IndexedOptionArray:
    """Strategy for [`ak.contents.IndexedOptionArray`][] instances within a size budget.

    Draws the index length first, then gives the remainder of the budget
    to the child content.

    Called by `contents()` during recursive tree generation.

    Parameters
    ----------
    content
        A callable that accepts `max_size` and `max_leaf_size` and returns
        a strategy for a single content.
    max_size
        Upper bound on `content_size()` of the result.
    max_leaf_size
        Upper bound on total leaf elements. Unbounded if `None`.
    min_length
        Lower bound on `len(result)`. Forwarded to `min_size` of the wrapper.
    max_length
        Upper bound on `len(result)`. Unbounded if `None`.
    st_option
        Accepted for `_StFromContents` compatibility; unused in this variant.

    Returns
    -------
    IndexedOptionArray

    Examples
    --------
    >>> from hypothesis_awkward.util import content_size, leaf_size
    >>> contents = st_ak.contents.contents
    >>> c = indexed_option_array_from_contents(
    ...     contents, max_size=20, max_leaf_size=10, min_length=2, max_length=5
    ... ).example()
    >>> isinstance(c, IndexedOptionArray)
    True

    >>> content_size(c) <= 20
    True

    >>> leaf_size(c) <= 10
    True

    >>> 2 <= len(c) <= 5
    True
    """
    ml = max_length if max_length is not None else max_size
    assume(min_length <= ml)
    n = draw(st.integers(min_value=min_length, max_value=ml))
    max_content_size = max(max_size - n, 0)
    child = draw(
        content(
            max_size=max_content_size,
            max_leaf_size=max_leaf_size,
            allow_option_root=False,
            allow_union_root=False,
        )
    )
    result = draw(indexed_option_array_contents(child, min_size=n, max_size=n))
    assume(content_size(result) <= max_size)
    return result
