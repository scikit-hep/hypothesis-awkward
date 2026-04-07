from typing import TYPE_CHECKING

from hypothesis import assume
from hypothesis import strategies as st

import hypothesis_awkward.strategies as st_ak
from awkward.contents import Content, RegularArray
from hypothesis_awkward.util.awkward import content_size

if TYPE_CHECKING:
    from .content import StContent
    from .option import StOption


@st.composite
def regular_array_contents(
    draw: st.DrawFn,
    content: st.SearchStrategy[Content] | Content | None = None,
    *,
    max_size: int | None = None,
    max_zeros_length: int | None = None,
    max_length: int | None = None,
) -> RegularArray:
    """Strategy for ``RegularArray``.

    Parameters
    ----------
    content
        Child content. Can be a strategy for Content, a concrete Content instance, or
        ``None`` to draw from ``contents()``.
    max_size
        Upper bound on the length of each element. Defaults to ``len(content)``
        when ``None``.
    max_zeros_length
        Upper bound on the number of elements when each element is empty, i.e., when
        size is zero. Defaults to ``len(content)`` when ``None``.
    max_length
        Upper bound on the number of groups, i.e., ``len(result)``.

    Examples
    --------
    >>> c = regular_array_contents().example()
    >>> isinstance(c, Content)
    True

    Limit each element to at most 3 items:

    >>> c = regular_array_contents(max_size=3).example()
    >>> c.size <= 3
    True

    Limit the number of elements when size is zero:

    >>> c = regular_array_contents(max_size=0, max_zeros_length=2).example()
    >>> c.size == 0 and len(c) <= 2
    True

    Limit the number of groups:

    >>> c = regular_array_contents(max_length=4).example()
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
    if max_size is None:
        max_size = content_len
    if max_zeros_length is None:
        max_zeros_length = content_len
    size = draw(
        _st_group_sizes(content_len, max_group_size=max_size, max_length=max_length)
    )
    if size == 0:
        max_zl = max_zeros_length
        if max_length is not None:
            max_zl = min(max_zl, max_length)
        zeros_length = draw(st.integers(min_value=0, max_value=max_zl))
        return RegularArray(content, size=0, zeros_length=zeros_length)
    return RegularArray(content, size=size)


def _st_group_sizes(
    total_items: int,
    *,
    min_group_size: int = 0,
    max_group_size: int | None = None,
    max_length: int | None = None,
    allow_non_divisors: bool = True,
) -> st.SearchStrategy[int]:
    """Strategy for the size parameter of a RegularArray.

    A RegularArray subdivides ``total_items`` into equal groups of ``group_size``, so
    ``group_size`` must be a divisor of ``total_items`` and at most ``max_group_size``.

    When ``total_items == 0``, any group size up to ``max_group_size`` is valid because
    zero items can be split into zero groups of any size.

    When ``max_length`` is set, only divisors that produce at most ``max_length`` groups
    are considered, i.e., divisors ``d`` where ``total_items // d <= max_length``.

    When ``total_items > 0`` but no valid divisor exists (i.e., ``max_group_size == 0``
    or ``max_length`` is too small), returns ``0``. The caller uses this to fall back to
    the ``zeros_length`` path, producing a RegularArray whose elements are all empty.

    Parameters
    ----------
    total_items
        Total number of items in the content.
    min_group_size
        Lower bound on the group size. Defaults to ``0``.
    max_group_size
        Upper bound on the group size. ``None`` means no constraint beyond
        ``total_items``.
    allow_non_divisors
        When ``False`` (the default), only divisors of ``total_items`` are
        returned. When ``True``, non-divisors are also included but listed after
        divisors so that shrinking prefers divisors (no unreachable data).
    max_length
        Upper bound on the number of groups. ``None`` means no constraint.
    """
    if max_group_size is None:
        max_group_size = total_items
    if total_items == 0:
        return st.integers(min_value=min_group_size, max_value=max_group_size)
    max_group_size = min(total_items, max_group_size)
    if max_length is not None and max_length == 0:
        return st.just(0)
    effective_min = max(min_group_size, 1)
    if max_length is not None:
        effective_min = max(effective_min, -(-total_items // max_length))
    divisors = [
        d for d in range(max_group_size, effective_min - 1, -1) if total_items % d == 0
    ]
    if not allow_non_divisors:
        group_sizes = divisors
    else:
        non_divisors = [
            d
            for d in range(max_group_size, effective_min - 1, -1)
            if total_items % d != 0
        ]
        group_sizes = divisors + non_divisors
    if not group_sizes:
        return st.just(0)
    return st.sampled_from(group_sizes)


@st.composite
def regular_array_from_contents(
    draw: st.DrawFn,
    content: 'StContent',
    *,
    max_size: int,
    max_leaf_size: 'int | None' = None,
    max_length: 'int | None' = None,
    st_option: 'StOption | None' = None,
) -> RegularArray:
    """Strategy for inner ``RegularArray`` to be drawn by an outer layout strategy.

    This strategy is called by an outer layout strategy. The argument ``content`` is a
    function that returns a strategy for the inner layout of the ``RegularArray``.

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
        Upper bound on the number of groups, i.e., ``len(result)``.

    Examples
    --------
    >>> from hypothesis_awkward.util.awkward import content_size, leaf_size
    >>> contents = st_ak.contents.contents
    >>> c = regular_array_from_contents(
    ...     contents, max_size=20, max_leaf_size=10, max_length=5
    ... ).example()
    >>> isinstance(c, Content)
    True

    >>> content_size(c) <= 20
    True

    >>> leaf_size(c) <= 10
    True

    >>> len(c) <= 5
    True
    """
    max_content_size = max(max_size - 1, 0)
    st_content = content(max_size=max_content_size, max_leaf_size=max_leaf_size)
    result = draw(regular_array_contents(st_content, max_length=max_length))
    assume(content_size(result) <= max_size)
    return result
