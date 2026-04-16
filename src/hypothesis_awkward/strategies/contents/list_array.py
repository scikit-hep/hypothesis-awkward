from typing import TYPE_CHECKING

from hypothesis import reject
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak
from awkward.contents import Content, ListArray
from hypothesis_awkward.util.safe import safe_min

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
    """Strategy for [`ak.contents.ListArray`][] instances.

    This strategy generates a [`ListArray`][ak.contents.ListArray] with the given content. It produces all
    valid [`ListArray`][ak.contents.ListArray] layouts including unreachable data, gaps, overlapping sublists,
    and out-of-order starts. It shrinks toward contiguous, monotonic starts with no
    unreachable data.

    Parameters
    ----------
    content
        Child content. Can be a strategy for [`Content`][ak.contents.Content], a concrete [`Content`][ak.contents.Content] instance, or
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
    starts_list, stops_list = draw(
        _st_starts_stops(len(content), max_length=max_length)
    )
    starts = ak.index.Index64(starts_list)
    stops = ak.index.Index64(stops_list)
    return ListArray(starts, stops, content)


@st.composite
def _st_starts_stops(
    draw: st.DrawFn,
    content_len: int,
    *,
    max_length: int | None = None,
    allow_unreachable: bool = True,
) -> tuple[list[int], list[int]]:
    """Strategy for starts and stops of a [`ListArray`][ak.contents.ListArray].

    Shrinks toward no unreachable data (``starts[0] == 0`` and
    ``stops[-1] == content_len``).

    Parameters
    ----------
    content_len
        Length of the content array.
    max_length
        Upper bound on the number of lists (i.e., ``len(starts)``).
    allow_unreachable
        No unreachable data is possible if ``False``.
    """
    if content_len == 0 or not allow_unreachable:
        return draw(_st_starts_stops_no_unreachable(content_len, max_length=max_length))
    if max_length is not None and max_length == 0:
        return [], []
    branches = [
        _st_starts_stops_no_unreachable(content_len, max_length=max_length),
        _st_starts_stops_unreachable(content_len, max_length=max_length),
    ]
    ml = max_length if max_length is not None else content_len
    if ml >= 2:
        branches.append(_st_starts_stops_gaps(content_len, max_length=max_length))
        if content_len >= 2:
            branches.append(
                _st_starts_stops_overlapping(content_len, max_length=max_length)
            )
        branches.append(
            _st_starts_stops_out_of_order(content_len, max_length=max_length)
        )
    return draw(st.one_of(branches))


@st.composite
def _st_starts_stops_no_unreachable(
    draw: st.DrawFn,
    content_len: int,
    *,
    max_length: int | None = None,
) -> tuple[list[int], list[int]]:
    """Strategy for starts and stops with no unreachable data."""
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
    return offsets_list[:-1], offsets_list[1:]


@st.composite
def _st_starts_stops_unreachable(
    draw: st.DrawFn,
    content_len: int,
    *,
    max_length: int | None = None,
) -> tuple[list[int], list[int]]:
    """Strategy for starts and stops with unreachable data.

    Guarantees at least unreachable tail.
    """
    max_size = None if max_length is None else max_length + 1
    offsets_list = sorted(
        draw(
            st.lists(
                st.integers(min_value=0, max_value=content_len - 1),
                min_size=2,
                max_size=max_size,
            )
        )
    )
    return offsets_list[:-1], offsets_list[1:]


@st.composite
def _st_starts_stops_gaps(
    draw: st.DrawFn,
    content_len: int,
    *,
    max_length: int | None = None,
) -> tuple[list[int], list[int]]:
    """Strategy for starts and stops with at least one gap between sublists.

    Guarantees at least one ``stops[i] < starts[i + 1]``.
    """
    ml = max_length if max_length is not None else content_len
    n = draw(st.integers(min_value=2, max_value=ml))
    values = sorted(
        draw(
            st.lists(
                st.integers(min_value=0, max_value=content_len),
                min_size=2 * n,
                max_size=2 * n,
            )
        )
    )
    starts = [values[2 * i] for i in range(n)]
    stops = [values[2 * i + 1] for i in range(n)]
    if not any(stops[i] < starts[i + 1] for i in range(n - 1)):
        reject()
    return starts, stops


@st.composite
def _st_starts_stops_overlapping(
    draw: st.DrawFn,
    content_len: int,
    *,
    max_length: int | None = None,
) -> tuple[list[int], list[int]]:
    """Strategy for starts and stops with at least one overlapping pair.

    Guarantees at least one ``starts[i + 1] < stops[i]``.
    """
    ml = max_length if max_length is not None else content_len
    n = draw(st.integers(min_value=2, max_value=ml))
    starts = sorted(
        draw(
            st.lists(
                st.integers(min_value=0, max_value=content_len),
                min_size=n,
                max_size=n,
            )
        )
    )
    stops = [
        draw(st.integers(min_value=starts[i], max_value=content_len)) for i in range(n)
    ]
    if not any(starts[i + 1] < stops[i] for i in range(n - 1)):
        reject()
    return starts, stops


@st.composite
def _st_starts_stops_out_of_order(
    draw: st.DrawFn,
    content_len: int,
    *,
    max_length: int | None = None,
) -> tuple[list[int], list[int]]:
    """Strategy for starts and stops with non-monotonic starts.

    Guarantees at least one ``starts[i] > starts[i + 1]``.
    """
    ml = max_length if max_length is not None else content_len
    n = draw(st.integers(min_value=2, max_value=ml))
    pairs = [
        draw(
            st.tuples(st.integers(0, content_len), st.integers(0, content_len)).filter(
                lambda p: p[0] <= p[1]
            )
        )
        for _ in range(n)
    ]
    starts = [p[0] for p in pairs]
    stops = [p[1] for p in pairs]
    if not any(starts[i] > starts[i + 1] for i in range(n - 1)):
        reject()
    return starts, stops


@st.composite
def list_array_from_contents(
    draw: st.DrawFn,
    content: 'StContent',
    *,
    max_size: int,
    max_leaf_size: 'int | None' = None,
    max_length: 'int | None' = None,
    st_option: 'StOption | None' = None,
) -> ListArray:
    """Strategy for inner [`ak.contents.ListArray`][] within an outer layout.

    This strategy is called by an outer layout strategy. The argument ``content`` is a
    function that returns a strategy for the inner layout of the [`ListArray`][ak.contents.ListArray].

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
        Upper bound on ``len(result)``, i.e., ``len(result.starts) = len(result.stops)``.

    Examples
    --------
    >>> from hypothesis_awkward.util.awkward import content_size, leaf_size
    >>> contents = st_ak.contents.contents
    >>> c = list_array_from_contents(
    ...     contents, max_size=20, max_leaf_size=10, max_length=5
    ... ).example()
    >>> isinstance(c, ListArray)
    True

    >>> content_size(c) <= 20
    True

    >>> leaf_size(c) <= 10
    True

    >>> len(c) <= 5
    True
    """
    max_length = safe_min((max_length, max_size // 2))
    length = draw(st.integers(min_value=0, max_value=max_length))
    indices_size = 2 * length  # starts and stops
    max_content_size = max_size - indices_size
    st_content = content(max_size=max_content_size, max_leaf_size=max_leaf_size)

    # TODO: Add `min_length=length` when `min_length` is implemented.
    return draw(list_array_contents(st_content, max_length=length))
