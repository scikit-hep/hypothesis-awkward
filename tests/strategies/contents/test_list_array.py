from typing import TypedDict, cast

from hypothesis import given, settings
from hypothesis import strategies as st

import hypothesis_awkward.strategies as st_ak
from awkward.contents import Content, ListArray

MAX_LIST_LENGTH = 5


class ListArrayContentsKwargs(TypedDict, total=False):
    '''Options for `list_array_contents()` strategy.'''

    content: st.SearchStrategy[Content] | Content


def list_array_contents_kwargs() -> st.SearchStrategy[
    st_ak.Opts[ListArrayContentsKwargs]
]:
    '''Strategy for options for `list_array_contents()` strategy.'''
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
        .map(lambda d: cast(ListArrayContentsKwargs, d))
        .map(st_ak.Opts[ListArrayContentsKwargs])
    )


@settings(max_examples=200)
@given(data=st.data())
def test_list_array_contents(data: st.DataObject) -> None:
    '''Test that `list_array_contents()` respects all its options.'''
    opts = data.draw(list_array_contents_kwargs(), label='opts')
    opts.reset()
    result = data.draw(
        st_ak.contents.list_array_contents(**opts.kwargs), label='result'
    )

    # Assert the result is always a ListArray content
    assert isinstance(result, ListArray)

    # Assert list length is within bounds
    assert 0 <= len(result) <= MAX_LIST_LENGTH

    # Assert starts and stops have the same length
    starts = result.starts.data
    stops = result.stops.data
    assert len(starts) == len(stops)

    # Assert starts[i] <= stops[i] and stops do not exceed content length
    for i in range(len(starts)):
        assert starts[i] <= stops[i]
        assert stops[i] <= len(result.content)

    # Assert content
    content = opts.kwargs.get('content', None)
    match content:
        case Content():
            assert result.content is content
        case st_ak.RecordDraws():
            assert len(content.drawn) == 1
            assert result.content is content.drawn[0]
