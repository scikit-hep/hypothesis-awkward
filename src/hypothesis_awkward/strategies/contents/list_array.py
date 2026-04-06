from typing import TYPE_CHECKING

import numpy as np
from hypothesis import assume
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak
from awkward.contents import Content, ListArray
from hypothesis_awkward.util.awkward import content_size

if TYPE_CHECKING:
    from .content import StContent
    from .option import StOption


@st.composite
def list_array_contents(
    draw: st.DrawFn,
    content: st.SearchStrategy[Content] | Content | None = None,
    *,
    max_length: int | None = None,
) -> ListArray:
    """Strategy for ListArray Content wrapping child Content.

    Parameters
    ----------
    content
        Child content. Can be a strategy for Content, a concrete Content instance, or
        ``None`` to draw from ``contents()``.
    max_length
        Upper bound on the number of lists, i.e., ``len(result)``. Defaults
        to ``len(content)`` when ``None``.

    Examples
    --------
    >>> c = list_array_contents().example()
    >>> isinstance(c, Content)
    True

    Limit the number of lists:

    >>> c = list_array_contents(max_length=4).example()
    >>> len(c) <= 4
    True
    """
    match content:
        case None:
            content = draw(st_ak.contents.contents())
        case st.SearchStrategy():
            content = draw(content)
        case Content():
            pass
    assert isinstance(content, Content)
    content_len = len(content)
    ml = max_length if max_length is not None else content_len
    n = draw(st.integers(min_value=0, max_value=ml))
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
    starts = ak.index.Index64(offsets[:-1])
    stops = ak.index.Index64(offsets[1:])
    return ListArray(starts, stops, content)


@st.composite
def list_array_from_contents(
    draw: st.DrawFn,
    content: 'StContent',
    *,
    max_size: int,
    max_leaf_size: 'int | None',
    max_length: 'int | None' = None,
    st_option: 'StOption | None' = None,
) -> ListArray:
    """Strategy that generates a starts/stops list layout within a size limit.

    Draws the number of lists ``n`` first and computes the starts/stops
    overhead (``2 * n``). The remainder of ``max_size`` after this deduction
    is passed to ``content`` to generate the inner content.

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
        to ``max_size // 2`` when ``None``.
    """
    ml = max_length if max_length is not None else max_size // 2
    n = draw(st.integers(min_value=0, max_value=ml))
    overhead = 2 * n
    max_content_size = max(max_size - overhead, 0)
    st_content = content(max_size=max_content_size, max_leaf_size=max_leaf_size)
    result = draw(list_array_contents(st_content, max_length=n))
    assume(content_size(result) <= max_size)
    return result
