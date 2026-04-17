from typing import TYPE_CHECKING

from hypothesis import strategies as st

from awkward.contents import Content, UnmaskedArray
from hypothesis_awkward import strategies as st_ak

if TYPE_CHECKING:
    from .content import StContent
    from .option import StOption


@st.composite
def unmasked_array_contents(
    draw: st.DrawFn,
    content: st.SearchStrategy[Content] | Content | None = None,
) -> UnmaskedArray:
    """Strategy for [`ak.contents.UnmaskedArray`][] instances.

    [`UnmaskedArray`][ak.contents.UnmaskedArray] is an option type with no actual nulls.
    It simply wraps content, adding option-type semantics without a mask buffer.

    Parameters
    ----------
    content
        Child content. Can be a strategy for [`Content`][ak.contents.Content], a concrete
        [`Content`][ak.contents.Content] instance, or ``None`` to draw from
        ``contents()``.

    Returns
    -------
    UnmaskedArray

    Examples
    --------
    >>> c = unmasked_array_contents().example()
    >>> isinstance(c, UnmaskedArray)
    True
    """
    match content:
        case None:
            content = draw(
                st_ak.contents.contents(allow_union_root=False, allow_option_root=False)
            )
        case st.SearchStrategy():
            content = draw(content)
        case Content():
            pass
    assert isinstance(content, Content)
    return UnmaskedArray(content)


@st.composite
def unmasked_array_from_contents(
    draw: st.DrawFn,
    content: 'StContent',
    *,
    max_size: int,
    max_leaf_size: int | None = None,
    max_length: int | None = None,
    st_option: 'StOption | None' = None,
) -> UnmaskedArray:
    """Strategy for [`ak.contents.UnmaskedArray`][] instances within a size budget.

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

    Returns
    -------
    UnmaskedArray
    """
    max_content_size = max_size
    if max_length is not None:
        max_content_size = min(max_content_size, max_length)
    child = draw(
        content(
            max_size=max_content_size,
            max_leaf_size=max_leaf_size,
            allow_option_root=False,
            allow_union_root=False,
        )
    )
    return UnmaskedArray(child)
