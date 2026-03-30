from hypothesis import strategies as st

import hypothesis_awkward.strategies as st_ak
from awkward.contents import Content, UnmaskedArray


@st.composite
def unmasked_array_contents(
    draw: st.DrawFn,
    content: st.SearchStrategy[Content] | Content | None = None,
) -> UnmaskedArray:
    '''Strategy for UnmaskedArray Content wrapping child Content.

    ``UnmaskedArray`` is an option type with no actual nulls. It simply
    wraps content, adding option-type semantics without a mask buffer.

    Parameters
    ----------
    content
        Child content. Can be a strategy for Content, a concrete Content instance, or
        ``None`` to draw from ``contents()``.

    Examples
    --------
    >>> c = unmasked_array_contents().example()
    >>> isinstance(c, Content)
    True
    '''
    match content:
        case None:
            content = draw(st_ak.contents.contents(allow_union_root=False))
        case st.SearchStrategy():
            content = draw(content)
        case Content():
            pass
    assert isinstance(content, Content)
    return UnmaskedArray(content)
