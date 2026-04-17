import functools
from typing import TYPE_CHECKING

import numpy as np
from hypothesis import assume
from hypothesis import strategies as st

import awkward as ak
from awkward.contents import Content, UnionArray
from hypothesis_awkward import strategies as st_ak
from hypothesis_awkward.util import content_size

from .option import option_from_contents

if TYPE_CHECKING:
    from .content import StContent
    from .option import StOption


@st.composite
def union_array_contents(
    draw: st.DrawFn,
    contents: list[Content] | st.SearchStrategy[list[Content]] | None = None,
    *,
    max_contents: int = 4,
    max_length: int | None = None,
) -> UnionArray:
    """Strategy for [`ak.contents.UnionArray`][] instances.

    Parameters
    ----------
    contents
        Child contents. Can be a strategy for a list of [`Content`][ak.contents.Content],
        a concrete list, or ``None`` to draw random children.
    max_contents
        Maximum number of child contents when ``contents`` is ``None``.
    max_length
        Upper bound on the union length, i.e., ``len(result)``. Unbounded if
        ``None``.

    Returns
    -------
    UnionArray

    Examples
    --------
    >>> c = union_array_contents().example()
    >>> isinstance(c, UnionArray)
    True

    Limit the union length:

    >>> c = union_array_contents(max_length=4).example()
    >>> len(c) <= 4
    True
    """
    match contents:
        case None:
            contents = draw(
                st_ak.contents.content_lists(
                    functools.partial(
                        st_ak.contents.contents,
                        allow_union_root=False,
                    ),
                    max_leaf_size=max_length if max_length is not None else 10,
                    min_len=2,
                    max_len=max_contents,
                    all_option_or_none=True,
                    st_option=option_from_contents,
                )
            )
            if max_length is not None:
                assume(sum(len(c) for c in contents) <= max_length)
        case st.SearchStrategy():
            contents = draw(contents)
        case list():
            pass
    assert isinstance(contents, list)

    # Build tags and index arrays
    tags_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    for k, content in enumerate(contents):
        length = len(content)
        tags_parts.append(np.full(length, k, dtype=np.int8))
        index_parts.append(np.arange(length, dtype=np.int64))

    if tags_parts:
        tags_flat = np.concatenate(tags_parts)
        index_flat = np.concatenate(index_parts)

        # Shuffle to interleave contents
        perm = draw(st.permutations(range(len(tags_flat))))
        tags_flat = tags_flat[list(perm)]
        index_flat = index_flat[list(perm)]

        # Truncate if concrete/strategy contents exceed max_length
        if max_length is not None and len(tags_flat) > max_length:
            tags_flat = tags_flat[:max_length]
            index_flat = index_flat[:max_length]
    else:
        tags_flat = np.array([], dtype=np.int8)
        index_flat = np.array([], dtype=np.int64)

    return UnionArray(
        tags=ak.index.Index8(tags_flat),
        index=ak.index.Index64(index_flat),
        contents=contents,
    )


@st.composite
def union_array_from_contents(
    draw: st.DrawFn,
    content: 'StContent',
    *,
    max_size: int,
    max_leaf_size: int | None = None,
    max_length: int | None = None,
    st_option: 'StOption | None' = None,
) -> UnionArray:
    """Strategy for [`ak.contents.UnionArray`][] instances within a size budget.

    Draws multiple children via ``content_lists()`` with ``min_len=2``, then wraps
    them in a [`UnionArray`][ak.contents.UnionArray] with shuffled tags and index
    arrays. Prevents nested unions by passing ``allow_union_root=False`` to child
    generation.
    Enforces the all-or-none option rule via ``all_option_or_none=True``.

    Called by ``contents()`` during recursive tree generation.

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
    st_option
        A callable conforming to ``StOption`` that wraps content in an option
        type. Used for all-or-none option coordination.

    Returns
    -------
    UnionArray

    Examples
    --------
    >>> from hypothesis_awkward.util import content_size, leaf_size
    >>> contents = st_ak.contents.contents
    >>> c = union_array_from_contents(
    ...     contents,
    ...     max_size=20,
    ...     max_leaf_size=10,
    ...     max_length=5,
    ...     st_option=option_from_contents,
    ... ).example()
    >>> isinstance(c, UnionArray)
    True

    >>> content_size(c) <= 20
    True

    >>> leaf_size(c) <= 10
    True

    >>> len(c) <= 5
    True
    """
    children = draw(
        st_ak.contents.content_lists(
            functools.partial(content, allow_union_root=False),
            max_size=max_size,
            max_leaf_size=max_leaf_size,
            min_len=2,
            all_option_or_none=st_option is not None,
            st_option=st_option,
        )
    )
    result = draw(union_array_contents(children, max_length=max_length))
    assume(content_size(result) <= max_size)
    return result
