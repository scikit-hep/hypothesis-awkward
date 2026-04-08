from typing import TYPE_CHECKING

import numpy as np
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak
from awkward.contents import Content, ListOffsetArray
from hypothesis_awkward.util.safe import safe_min

if TYPE_CHECKING:
    from .content import StContent
    from .option import StOption


@st.composite
def list_offset_array_contents(
    draw: st.DrawFn,
    content: st.SearchStrategy[Content] | Content | None = None,
    *,
    max_length: int | None = None,
) -> ListOffsetArray:
    """Strategy for ``ListOffsetArray``.

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
    >>> c = list_offset_array_contents().example()
    >>> isinstance(c, Content)
    True

    Limit the number of lists:

    >>> c = list_offset_array_contents(max_length=4).example()
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
    return ListOffsetArray(ak.index.Index64(offsets), content)


@st.composite
def list_offset_array_from_contents(
    draw: st.DrawFn,
    content: 'StContent',
    *,
    max_size: int,
    max_leaf_size: 'int | None' = None,
    max_length: 'int | None' = None,
    st_option: 'StOption | None' = None,
) -> ListOffsetArray:
    """Strategy for inner ``ListOffsetArray`` to be drawn by an outer layout strategy.

    This strategy is called by an outer layout strategy. The argument ``content`` is a
    function that returns a strategy for the inner layout of the ``ListOffsetArray``.

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
        Upper bound on ``len(result)``, i.e., ``len(result.offsets) - 1``.

    Examples
    --------
    >>> from hypothesis_awkward.util.awkward import content_size, leaf_size
    >>> contents = st_ak.contents.contents
    >>> c = list_offset_array_from_contents(
    ...     contents, max_size=20, max_leaf_size=10, max_length=5
    ... ).example()
    >>> isinstance(c, ListOffsetArray)
    True

    >>> content_size(c) <= 20
    True

    >>> leaf_size(c) <= 10
    True

    >>> len(c) <= 5
    True
    """
    max_length = safe_min((max_length, max_size - 1))
    length = draw(st.integers(min_value=0, max_value=max_length))
    offsets_size = length + 1
    max_content_size = max_size - offsets_size
    st_content = content(max_size=max_content_size, max_leaf_size=max_leaf_size)

    # TODO: Add `min_length=length` when `min_length` is implemented.
    return draw(list_offset_array_contents(st_content, max_length=length))
