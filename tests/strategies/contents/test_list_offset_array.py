from typing import TypedDict, cast

from hypothesis import given, settings
from hypothesis import strategies as st

import hypothesis_awkward.strategies as st_ak
from awkward.contents import Content, ListOffsetArray

MAX_LIST_LENGTH = 5


class ListOffsetArrayContentsKwargs(TypedDict, total=False):
    '''Options for `list_offset_array_contents()` strategy.'''

    content: st.SearchStrategy[Content] | Content


def list_offset_array_contents_kwargs() -> st.SearchStrategy[
    st_ak.Opts[ListOffsetArrayContentsKwargs]
]:
    '''Strategy for options for `list_offset_array_contents()` strategy.'''
    return (
        st.fixed_dictionaries(
            {},
            optional={
                'content': st.one_of(
                    st_ak.contents.contents(),
                    st.just(
                        st_ak.RecordDraws(st_ak.contents.contents())
                    ),
                ),
            },
        )
        .map(lambda d: cast(ListOffsetArrayContentsKwargs, d))
        .map(st_ak.Opts[ListOffsetArrayContentsKwargs])
    )


@settings(max_examples=200)
@given(data=st.data())
def test_list_offset_array_contents(data: st.DataObject) -> None:
    '''Test that `list_offset_array_contents()` respects all its options.'''
    opts = data.draw(list_offset_array_contents_kwargs(), label='opts')
    opts.reset()
    result = data.draw(
        st_ak.contents.list_offset_array_contents(**opts.kwargs), label='result'
    )

    # Assert the result is always a ListOffsetArray content
    assert isinstance(result, ListOffsetArray)

    # Assert list length is within bounds
    assert 0 <= len(result) <= MAX_LIST_LENGTH

    # Assert offsets are monotonically non-decreasing
    offsets = result.offsets.data
    for i in range(len(offsets) - 1):
        assert offsets[i] <= offsets[i + 1]

    # Assert first offset is 0 and last offset does not exceed content length
    assert offsets[0] == 0
    assert offsets[-1] <= len(result.content)

    # Assert content
    content = opts.kwargs.get('content', None)
    match content:
        case Content():
            assert result.content is content
        case st_ak.RecordDraws():
            assert len(content.drawn) == 1
            assert result.content is content.drawn[0]
