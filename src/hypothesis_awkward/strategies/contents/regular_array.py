from hypothesis import strategies as st

import hypothesis_awkward.strategies as st_ak
from awkward.contents import Content, RegularArray


@st.composite
def regular_array_contents(
    draw: st.DrawFn,
    content: st.SearchStrategy[Content] | Content | None = None,
    *,
    max_size: int = 5,
    max_zeros_length: int = 5,
) -> Content:
    '''Strategy for RegularArray Content wrapping child Content.

    Parameters
    ----------
    content
        Child content. Can be a strategy for Content, a concrete Content
        instance, or ``None`` to draw from ``contents()``.
    max_size
        Upper bound on the length of each element.
    max_zeros_length
        Upper bound on the number of elements when each element is
        empty, i.e., when size is zero.

    Examples
    --------
    >>> c = regular_array_contents().example()
    >>> isinstance(c, Content)
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
    size, zeros_length = draw(
        _size_and_zeros_length(len(content), max_size, max_zeros_length)
    )
    return RegularArray(content, size=size, zeros_length=zeros_length)


@st.composite
def _size_and_zeros_length(
    draw: st.DrawFn,
    content_len: int,
    max_size: int,
    max_zeros_length: int,
) -> tuple[int, int]:
    if content_len == 0:
        size = draw(st.integers(min_value=0, max_value=max_size))
        if size == 0:
            zeros_length = draw(st.integers(min_value=0, max_value=max_zeros_length))
            return size, zeros_length
        return size, 0
    divisors = [
        d for d in range(1, min(content_len + 1, max_size + 1)) if content_len % d == 0
    ]
    if not divisors:
        zeros_length = draw(st.integers(min_value=0, max_value=max_zeros_length))
        return 0, zeros_length
    size = draw(st.sampled_from(divisors))
    return size, 0
