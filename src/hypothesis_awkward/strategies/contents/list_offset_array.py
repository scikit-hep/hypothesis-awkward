from typing import TYPE_CHECKING

import numpy as np
from hypothesis import assume
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak
from awkward.contents import Content, ListOffsetArray
from hypothesis_awkward.util.awkward import content_size

if TYPE_CHECKING:
    from .content import StContent


@st.composite
def list_offset_array_contents(
    draw: st.DrawFn,
    content: st.SearchStrategy[Content] | Content | None = None,
    *,
    max_length: int = 5,
) -> ListOffsetArray:
    '''Strategy for ListOffsetArray Content wrapping child Content.

    Parameters
    ----------
    content
        Child content. Can be a strategy for Content, a concrete Content instance, or
        ``None`` to draw from ``contents()``.
    max_length
        Upper bound on the number of lists, i.e., ``len(result)``.

    Examples
    --------
    >>> c = list_offset_array_contents().example()
    >>> isinstance(c, Content)
    True

    Limit the number of lists:

    >>> c = list_offset_array_contents(max_length=4).example()
    >>> len(c) <= 4
    True
    '''
    match content:
        case None:
            content = draw(st_ak.contents.contents())
        case st.SearchStrategy():
            content = draw(content)
        case Content():
            pass
    assert isinstance(content, Content)
    content_len = len(content)
    n = draw(st.integers(min_value=0, max_value=max_length))
    if n == 0:
        offsets_list = [0]
    elif content_len == 0:
        offsets_list = [0] * (n + 1)
    else:
        splits = sorted(
            draw(
                st.lists(
                    st.integers(min_value=0, max_value=content_len),
                    min_size=n - 1,
                    max_size=n - 1,
                )
            )
        )
        offsets_list = [0, *splits, content_len]
    offsets = np.array(offsets_list, dtype=np.int64)
    return ListOffsetArray(ak.index.Index64(offsets), content)


@st.composite
def list_offset_array_from_contents(
    draw: st.DrawFn,
    content: 'StContent',
    *,
    max_size: int,
    max_leaf_size: 'int | None',
    max_length: 'int | None' = None,
) -> ListOffsetArray:
    '''Strategy that generates a variable-length list layout within a size limit.

    Draws the number of lists ``n`` first and computes the offset array size
    (``n + 1``). The remainder of ``max_size`` after this deduction is passed
    to ``content`` to generate the inner content. The result is validated
    against ``max_size`` via ``assume()``.

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
        Upper bound on the number of lists, i.e., ``len(result)``. Defaults
        to ``max_size - 1`` when ``None``.

    '''
    ml = max_length if max_length is not None else max_size - 1
    n = draw(st.integers(min_value=0, max_value=ml))
    overhead = n + 1
    max_content_size = max(max_size - overhead, 0)
    st_content = content(max_size=max_content_size, max_leaf_size=max_leaf_size)
    result = draw(list_offset_array_contents(st_content, max_length=n))
    assume(content_size(result) <= max_size)
    return result
