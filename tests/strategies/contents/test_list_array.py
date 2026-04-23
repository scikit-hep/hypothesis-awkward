from typing import Any, TypedDict, cast

import numpy as np
import pytest
from hypothesis import Phase, find, given, settings
from hypothesis import strategies as st

from awkward.contents import Content, ListArray, NumpyArray
from hypothesis_awkward import strategies as st_ak
from hypothesis_awkward.util import iter_contents
from hypothesis_awkward.util import safe_compare as sc


class ListArrayContentsKwargs(TypedDict, total=False):
    """Options for `list_array_contents()` strategy."""

    content: st.SearchStrategy[Content] | Content
    max_length: int | None


@st.composite
def list_array_contents_kwargs(
    draw: st.DrawFn,
    chain: st_ak.OptsChain[Any] | None = None,
) -> st_ak.OptsChain[ListArrayContentsKwargs]:
    """Strategy for options for `list_array_contents()` strategy."""
    if chain is None:
        chain = st_ak.OptsChain({})
    st_content = chain.register(st_ak.contents.contents())

    kwargs = draw(
        st.fixed_dictionaries(
            {},
            optional={
                'content': st.one_of(
                    st_ak.contents.contents(),
                    st.just(st_content),
                ),
                'max_length': st.integers(min_value=0, max_value=50),
            },
        )
    )

    return chain.extend(cast(ListArrayContentsKwargs, kwargs))


@settings(max_examples=200)
@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """Assert the results of `list_array_contents()`."""
    # Draw options
    opts = data.draw(list_array_contents_kwargs(), label='opts')
    opts.reset()

    # Call the test subject
    result = data.draw(
        st_ak.contents.list_array_contents(**opts.kwargs), label='result'
    )

    assert isinstance(result, ListArray)

    # Assert the options were effective
    max_length = opts.kwargs.get('max_length')
    assert len(result) <= sc(max_length)

    # Assert starts and stops have the same length
    starts = result.starts.data
    stops = result.stops.data
    assert len(starts) == len(stops)

    # Assert starts[i] <= stops[i] and stops do not exceed content length
    for i in range(len(starts)):
        assert starts[i] <= stops[i]
        assert stops[i] <= len(result.content)

    # Assert the options were effective
    content = opts.kwargs.get('content', None)
    match content:
        case Content():
            assert result.content is content
        case st_ak.RecordDraws():
            assert len(content.drawn) == 1
            assert result.content is content.drawn[0]


@pytest.mark.parametrize('max_length', [1, 2, 10])
def test_draw_max_length(max_length: int) -> None:
    """Assert the length can reach `max_length`."""
    find(
        st_ak.contents.list_array_contents(max_length=max_length),
        lambda c: len(c) == max_length,
    )


@pytest.mark.parametrize('len_content', [0, 1, 2, 10])
def test_draw_default_max_length(len_content: int) -> None:
    """Assert that len(result) can reach len(content) by default."""
    content = NumpyArray(np.zeros(len_content))
    assert len(content) == len_content
    find(
        st_ak.contents.list_array_contents(content),
        lambda c: len(c) == len(content),
    )


def test_draw_unreachable() -> None:
    """Assert data can be unreachable."""
    content = NumpyArray(np.arange(10))
    find(
        st_ak.contents.list_array_contents(content),
        lambda c: len(c) >= 1 and (c.starts[0] > 0 or c.stops[-1] < len(c.content)),
    )


@pytest.mark.xfail(reason='shrinker does not reliably reach no-unreachable layout')
def test_shrink_no_unreachable() -> None:
    """Assert reachable data only is the simplest."""
    content = NumpyArray(np.arange(10))
    c = find(
        st_ak.contents.list_array_contents(content),
        lambda c: len(c) >= 2,
    )
    assert c.starts[0] == 0
    assert c.stops[-1] == len(c.content)


def test_shrink_content_len_zero() -> None:
    """Assert no sublists are the simplest for an empty content."""
    content = NumpyArray(np.array([], dtype=np.int64))
    c = find(st_ak.contents.list_array_contents(content), lambda c: True)
    assert len(c) == 0


def test_draw_from_contents() -> None:
    """Assert `contents()` can generate a `ListArray` as outermost."""
    find(st_ak.contents.contents(), lambda c: isinstance(c, ListArray))


def test_draw_from_contents_variable_length() -> None:
    """Assert that ListArray with variable-length sublists can be drawn."""

    def _has_variable_length(c: Content) -> bool:
        return any(
            isinstance(n, ListArray)
            and len(n) >= 2
            and len(set(len(n[i]) for i in range(len(n)))) > 1
            for n in iter_contents(c)
        )

    find(
        st_ak.contents.contents(),
        _has_variable_length,
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_from_contents_empty_sublist() -> None:
    """Assert that ListArray with empty sublists can be drawn from `contents()`."""

    def _has_empty_sublist(c: Content) -> bool:
        return any(
            isinstance(n, ListArray) and any(len(n[i]) == 0 for i in range(len(n)))
            for n in iter_contents(c)
        )

    find(
        st_ak.contents.contents(),
        _has_empty_sublist,
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )
