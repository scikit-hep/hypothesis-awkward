import numpy as np
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak
from awkward.contents import Content, ListOffsetArray

MAX_LIST_LENGTH = 5


@st.composite
def list_offset_array_contents(
    draw: st.DrawFn,
    content: st.SearchStrategy[Content] | Content | None = None,
) -> Content:
    '''Strategy for ListOffsetArray Content wrapping child Content.'''
    if content is None:
        content = st_ak.contents.contents()
    content = draw(content)
    assert isinstance(content, Content)
    content_len = len(content)
    n = draw(st.integers(min_value=0, max_value=MAX_LIST_LENGTH))
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
    offsets = np.array(offsets_list, dtype=np.int64)
    return ListOffsetArray(ak.index.Index64(offsets), content)
