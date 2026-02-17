import string

from hypothesis import strategies as st

import hypothesis_awkward.strategies as st_ak
from awkward.contents import Content, RecordArray


@st.composite
def record_array_contents(
    draw: st.DrawFn,
    contents: list[Content] | st.SearchStrategy[list[Content]] | None = None,
    *,
    max_fields: int = 5,
    allow_tuple: bool = True,
) -> Content:
    '''Strategy for RecordArray Content from a list of child Contents.'''
    match contents:
        case None:
            n = draw(st.integers(min_value=0, max_value=max_fields))
            contents = [
                draw(st_ak.contents.contents()) for _ in range(n)
            ]
        case st.SearchStrategy():
            contents = draw(contents)
        case list():
            pass
    assert isinstance(contents, list)

    if allow_tuple:
        is_tuple = draw(st.booleans())
    else:
        is_tuple = False

    if is_tuple:
        fields = None
    else:
        st_names = st.text(alphabet=string.ascii_letters, max_size=3)
        fields = draw(
            st.lists(
                st_names,
                min_size=len(contents),
                max_size=len(contents),
                unique=True,
            )
        )

    length = 0 if not contents else None
    return RecordArray(contents, fields=fields, length=length)
