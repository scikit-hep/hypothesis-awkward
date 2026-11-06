import functools
from typing import TYPE_CHECKING

import numpy as np
from hypothesis import assume
from hypothesis import strategies as st

import awkward as ak
from awkward._do import mergeable
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
    min_length: int = 0,
    max_length: int | None = None,
    allow_invalid: bool = True,
) -> UnionArray:
    """Strategy for [`ak.contents.UnionArray`][] instances.

    Parameters
    ----------
    contents
        Child contents. Can be a strategy for a list of [`Content`][ak.contents.Content],
        a concrete list, or `None` to draw random children.
    max_contents
        Maximum number of child contents when `contents` is `None`. Below 2, the
        drawn contents cannot satisfy the minimum of
        [`UnionArray`][ak.contents.UnionArray], whose `TypeError` propagates.
    min_length
        Lower bound on the union length, i.e., `len(result)`. Constrains the
        pre-truncation sum of children lengths; never satisfied by truncation.
    max_length
        Upper bound on the union length, i.e., `len(result)`. Unbounded if
        `None`.
    allow_invalid
        Mergeable child contents can occur if `True`, which [`ak.validity_error`][]
        and [`ak.is_valid`][] consider invalid. No two child contents are mergeable
        if `False`: a concrete `contents` with mergeable ones raises `ValueError`,
        and such draws from a strategy are rejected. If `contents` is `None`, the
        child contents are also valid. Given child contents are not otherwise checked.

    Raises
    ------
    TypeError
        If fewer than two child contents result, whether from `max_contents` or
        from `contents`. Raised by [`UnionArray`][ak.contents.UnionArray].
    ValueError
        If `allow_invalid` is `False` and a concrete `contents` has mergeable
        child contents.

    Returns
    -------
    UnionArray

    Examples
    --------
    >>> c = union_array_contents().example()
    >>> isinstance(c, UnionArray)
    True

    Limit the union length:

    >>> c = union_array_contents(min_length=2, max_length=4).example()
    >>> 2 <= len(c) <= 4
    True
    """
    match contents:
        case None:
            contents = draw(
                st_ak.contents.content_lists(
                    functools.partial(
                        st_ak.contents.contents,
                        allow_invalid=allow_invalid,
                        allow_union_root=False,
                        allow_indexed_root=False,
                    ),
                    max_leaf_size=max_length if max_length is not None else 10,
                    # The clamp keeps min_len <= max_len: content_lists()
                    # honors min_len over max_len, so an unclamped 2 would
                    # override max_contents instead of letting UnionArray
                    # reject it.
                    min_len=min(2, max_contents),
                    max_len=max_contents,
                    all_option_or_none=True,
                    st_option=option_from_contents,
                )
            )
            if not allow_invalid:
                kept = _non_mergeable(contents)
                # Fewer than two drawn is left for `UnionArray` to reject.
                assume(len(kept) >= min(2, len(contents)))
                contents = kept
            if max_length is not None:
                assume(sum(len(c) for c in contents) <= max_length)
        case st.SearchStrategy():
            contents = draw(contents)
            if not allow_invalid:
                assume(len(_non_mergeable(contents)) == len(contents))
        case list():
            if not allow_invalid and len(_non_mergeable(contents)) < len(contents):
                raise ValueError(f'mergeable child contents: {contents!r}')
    assert isinstance(contents, list)

    return draw(
        _union_from_children(contents, min_length=min_length, max_length=max_length)
    )


@st.composite
def union_array_from_contents(
    draw: st.DrawFn,
    content: 'StContent',
    *,
    max_size: int,
    max_leaf_size: int | None = None,
    min_length: int = 0,
    max_length: int | None = None,
    allow_invalid: bool = True,
    st_option: 'StOption | None' = None,
) -> UnionArray:
    """Strategy for [`ak.contents.UnionArray`][] instances within a size budget.

    Draws multiple children via `content_lists()` with `min_len=2`, then wraps
    them in a [`UnionArray`][ak.contents.UnionArray] with shuffled tags and index
    arrays. Prevents nested unions by passing `allow_union_root=False` to child
    generation.
    Enforces the all-or-none option rule via `all_option_or_none=True`.

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
        Lower bound on `len(result)`.
    max_length
        Upper bound on `len(result)`. Unbounded if `None`.
    allow_invalid
        Mergeable child contents can occur if `True`. No two child contents are
        mergeable if `False`.
    st_option
        A callable conforming to `StOption` that wraps content in an option
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
    ...     min_length=2,
    ...     max_length=5,
    ...     st_option=option_from_contents,
    ... ).example()
    >>> isinstance(c, UnionArray)
    True

    >>> content_size(c) <= 20
    True

    >>> leaf_size(c) <= 10
    True

    >>> 2 <= len(c) <= 5
    True
    """
    children = draw(
        st_ak.contents.content_lists(
            functools.partial(
                content, allow_union_root=False, allow_indexed_root=False
            ),
            max_size=max_size,
            max_leaf_size=max_leaf_size,
            min_len=2,
            all_option_or_none=st_option is not None,
            st_option=st_option,
        )
    )
    if not allow_invalid:
        children = _non_mergeable(children)
        assume(len(children) >= 2)
    result = draw(
        _union_from_children(children, min_length=min_length, max_length=max_length)
    )
    assume(content_size(result) <= max_size)
    return result


def _non_mergeable(children: list[Content]) -> list[Content]:
    """Return `children` without those that merge with an earlier one."""
    kept = list[Content]()
    for c in children:
        if not any(mergeable(k, c, mergebool=False) for k in kept):
            kept.append(c)
    return kept


@st.composite
def _union_from_children(
    draw: st.DrawFn,
    children: list[Content],
    *,
    min_length: int = 0,
    max_length: int | None = None,
) -> UnionArray:
    """Strategy for a [`ak.contents.UnionArray`][] of `children`.

    Each element of `children` is referred to once, in a shuffled order, before
    truncation to `max_length`. The draw is rejected if `children` are too short for
    `min_length`.
    """
    # A pre-truncation sum floor.
    assume(sum(len(c) for c in children) >= min_length)

    # Build tags and index arrays
    tags_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    for k, content in enumerate(children):
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

        # Truncate if the children exceed max_length
        if max_length is not None and len(tags_flat) > max_length:
            tags_flat = tags_flat[:max_length]
            index_flat = index_flat[:max_length]
    else:
        tags_flat = np.array([], dtype=np.int8)
        index_flat = np.array([], dtype=np.int64)

    return UnionArray(
        tags=ak.index.Index8(tags_flat),
        index=ak.index.Index64(index_flat),
        contents=children,
    )
