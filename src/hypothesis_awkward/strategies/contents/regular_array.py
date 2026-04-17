from typing import TYPE_CHECKING

from hypothesis import strategies as st

from awkward.contents import Content, RegularArray
from hypothesis_awkward import strategies as st_ak

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
    """Strategy for [`ak.contents.RegularArray`][] instances.

    This strategy generates a [`RegularArray`][ak.contents.RegularArray] with the given
    content. It shrinks toward a shorter length (larger ``size``) with no unreachable
    data.

    Parameters
    ----------
    content
        content or strategy for the content. If ``None``, draw from ``contents()``.
    max_size
        Upper bound on the size parameter of the
        [`RegularArray`][ak.contents.RegularArray]. If ``None``, ``len(content)`` is
        used.
    max_zeros_length
        Upper bound on the ``zeros_length`` parameter of the
        [`RegularArray`][ak.contents.RegularArray]. Only effective when size is zero.
    max_length
        Upper bound on the length of the [`RegularArray`][ak.contents.RegularArray]
        (i.e., ``len(result)``). Unbounded if ``None``.


    Returns
    -------
    RegularArray

    Examples
    --------
    >>> c = regular_array_contents().example()
    >>> isinstance(c, RegularArray)
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


@st.composite
def _st_group_sizes(
    draw: st.DrawFn,
    total_items: int,
    *,
    min_group_size: int = 0,
    max_group_size: int | None = None,
    max_length: int | None = None,
    allow_non_divisors: bool = True,
) -> int:
    """Strategy for the size parameter of a [`RegularArray`][ak.contents.RegularArray].

    This strategy generates the size parameter for a [`RegularArray`][ak.contents.RegularArray] given the total
    number of items in the content and various constraints. It shrinks toward the
    divisors of ``total_items`` (no unreachable data) and a fewer groups (larger size).
    In other words, it shrinks toward the largest divisor of ``total_items`` that
    satisfies the constraints.

    When ``total_items == 0``, any group size between ``min_group_size`` and
    ``max_group_size`` is valid as zero items can be split into zero groups of any size.

    Parameters
    ----------
    total_items
        Total number of items in the content.
    min_group_size
        Lower bound on the group size. Defaults to ``0``.
    max_group_size
        Upper bound on the group size. Unbounded beyond ``total_items`` if
        ``None``.
    max_length
        Upper bound on the number of groups. Unbounded if ``None``.
    allow_non_divisors
        No unreachable data is possible if ``False``.
    """
    if max_group_size is None:
        max_group_size = total_items

    if total_items == 0:
        # Any size is possible without unreachable data.
        return draw(st.integers(min_value=min_group_size, max_value=max_group_size))

    if max_length is not None and max_length == 0:
        return 0

    # Finalize the bounds for finite return values.
    max_group_size = min(total_items, max_group_size)
    min_group_size = max(min_group_size, 1)
    if max_length is not None:
        min_group_size = max(min_group_size, -(-total_items // max_length))

    # Reversed for shrinking toward larger sizes (fewer groups).
    all_sizes = range(max_group_size, min_group_size - 1, -1)
    if not all_sizes:
        return 0

    # No unreachable data when the size is a divisor.
    divisors = [d for d in all_sizes if total_items % d == 0]

    reachable_allowed = len(divisors) > 0
    unreachable_allowed = allow_non_divisors and len(divisors) != len(all_sizes)
    if not (reachable_allowed or unreachable_allowed):
        return 0

    reachable_only = reachable_allowed and not unreachable_allowed
    if reachable_only:
        return draw(st.sampled_from(divisors))

    non_divisors = [d for d in all_sizes if d not in divisors]

    unreachable_only = unreachable_allowed and not reachable_allowed
    if unreachable_only:
        return draw(st.sampled_from(non_divisors))

    # Instead of drawing from concatenated divisors and non-divisors, draw ``one_of`` to
    # introduce an explicit branch for shrinking toward reachable data.
    return draw(st.one_of(st.sampled_from(divisors), st.sampled_from(non_divisors)))


@st.composite
def regular_array_from_contents(
    draw: st.DrawFn,
    content: 'StContent',
    *,
    max_size: int,
    max_leaf_size: int | None = None,
    max_length: int | None = None,
    st_option: 'StOption | None' = None,
) -> RegularArray:
    """Strategy for inner [`ak.contents.RegularArray`][] within an outer layout.

    This strategy is called by an outer layout strategy. The argument ``content`` is a
    function that returns a strategy for the inner layout of the [`RegularArray`][ak.contents.RegularArray].

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
    RegularArray

    Examples
    --------
    >>> from hypothesis_awkward.util import content_size, leaf_size
    >>> contents = st_ak.contents.contents
    >>> c = regular_array_from_contents(
    ...     contents, max_size=20, max_leaf_size=10, max_length=5
    ... ).example()
    >>> isinstance(c, RegularArray)
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
    return draw(regular_array_contents(st_content, max_length=max_length))
