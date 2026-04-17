from typing import TYPE_CHECKING

import numpy as np
from hypothesis import strategies as st

import awkward as ak
from awkward.contents import Content, ListOffsetArray
from hypothesis_awkward import strategies as st_ak
from hypothesis_awkward.util import safe_min

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
    """Strategy for [`ak.contents.ListOffsetArray`][] instances.

    This strategy generates a [`ListOffsetArray`][ak.contents.ListOffsetArray] with the
    given content. It produces layouts with and without unreachable data. It shrinks
    toward no unreachable data (``offsets[0] == 0`` and ``offsets[-1] == len(content)``).

    Parameters
    ----------
    content
        Child content. Can be a strategy for [`Content`][ak.contents.Content], a concrete
        [`Content`][ak.contents.Content] instance, or ``None`` to draw from
        ``contents()``.
    max_length
        Upper bound on the number of lists, i.e., ``len(result)``. If ``None``,
        ``len(content)`` is used.

    Returns
    -------
    ListOffsetArray

    Examples
    --------
    >>> c = list_offset_array_contents().example()
    >>> isinstance(c, ListOffsetArray)
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
    offsets_list = draw(_st_offsets(content_len, max_length=max_length))
    offsets = np.array(offsets_list, dtype=np.int64)
    return ListOffsetArray(ak.index.Index64(offsets), content)


@st.composite
def _st_offsets(
    draw: st.DrawFn,
    content_len: int,
    *,
    max_length: int | None = None,
    allow_unreachable: bool = True,
) -> list[int]:
    """Strategy for offsets of a [`ListOffsetArray`][ak.contents.ListOffsetArray].

    Shrinks toward no unreachable data (``offsets[0] == 0`` and
    ``offsets[-1] == content_len``).

    Parameters
    ----------
    content_len
        Length of the content array.
    max_length
        Upper bound on the length of the [`ListOffsetArray`][ak.contents.ListOffsetArray] (i.e.,
        ``len(offsets) - 1``).
    allow_unreachable
        No unreachable data is possible if ``False``.
    """
    if content_len == 0 or not allow_unreachable:
        return draw(_st_offsets_no_unreachable(content_len, max_length=max_length))
    if max_length is not None and max_length == 0:
        return [draw(st.integers(min_value=0, max_value=content_len))]
    return draw(
        st.one_of(
            _st_offsets_no_unreachable(content_len, max_length=max_length),
            _st_offsets_unreachable(content_len, max_length=max_length),
        )
    )


@st.composite
def _st_offsets_no_unreachable(
    draw: st.DrawFn,
    content_len: int,
    *,
    max_length: int | None = None,
) -> list[int]:
    """Strategy for offsets with no unreachable data."""
    if content_len == 0:
        ml = None if max_length is None else max_length + 1
        return draw(st.lists(st.just(0), min_size=1, max_size=ml))
    if max_length is not None:
        if max_length == 0:
            return [0]
        if max_length == 1:
            return [0, content_len]
    max_size = None if max_length is None else max_length - 1
    middle = sorted(
        draw(
            st.lists(st.integers(min_value=0, max_value=content_len), max_size=max_size)
        )
    )
    return [0, *middle, content_len]


@st.composite
def _st_offsets_unreachable(
    draw: st.DrawFn,
    content_len: int,
    *,
    max_length: int | None = None,
) -> list[int]:
    """Strategy for offsets with unreachable data (at least unreachable tail)."""
    max_size = None if max_length is None else max_length + 1
    offsets = sorted(
        draw(
            st.lists(
                st.integers(min_value=0, max_value=content_len - 1),
                min_size=1,
                max_size=max_size,
            )
        )
    )
    return offsets


@st.composite
def list_offset_array_from_contents(
    draw: st.DrawFn,
    content: 'StContent',
    *,
    max_size: int,
    max_leaf_size: int | None = None,
    max_length: int | None = None,
    st_option: 'StOption | None' = None,
) -> ListOffsetArray:
    """Strategy for inner [`ak.contents.ListOffsetArray`][] within an outer layout.

    This strategy is called by an outer layout strategy. The argument ``content`` is a
    function that returns a strategy for the inner layout of the [`ListOffsetArray`][ak.contents.ListOffsetArray].

    Parameters
    ----------
    content
        A callable that accepts ``max_size`` and ``max_leaf_size`` and returns
        a strategy for a single content.
    max_size
        Upper bound on ``content_size()`` of the result.
    max_leaf_size
        Upper bound on total leaf elements. Unbounded if ``None``.
    max_length
        Upper bound on ``len(result)``. Unbounded if ``None``.

    Returns
    -------
    ListOffsetArray

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
