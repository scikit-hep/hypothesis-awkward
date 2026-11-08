from typing import TypedDict

import numpy as np
import pytest
from hypothesis import find, given
from hypothesis import strategies as st

from awkward.contents import Content, ListArray, NumpyArray
from hypothesis_awkward import strategies as st_ak
from hypothesis_awkward._util import iter_contents
from hypothesis_awkward._util import safe_compare as sc
from tests.find_settings import FIND, FIND_NO_SHRINK
from tests.funcs import KwargsDrawn, assert_kwargs_match_signature


class ListArrayContentsKwargs(TypedDict, total=False):
    """Options for `list_array_contents()` strategy."""

    content: st.SearchStrategy[Content] | Content | None
    min_length: int
    max_length: int | None


DEFAULTS = ListArrayContentsKwargs(
    content=None,
    min_length=0,
    max_length=None,
)


class ListArrayContentsDrawn(TypedDict):
    """Values drawn in advance for the options of `list_array_contents()`."""

    content: Content | None


class RelatedKwargs(TypedDict, total=False):
    """Options drawn together as an ordered pair (`min_length <= max_length`)."""

    min_length: int
    max_length: int | None


def test_kwargs_match_signature() -> None:
    """Assert the option declarations agree with `list_array_contents()`."""
    assert_kwargs_match_signature(
        func=st_ak.constructors.list_array_contents,
        kwargs_cls=ListArrayContentsKwargs,
        defaults=DEFAULTS,
        related_cls=RelatedKwargs,
    )


@st.composite
def list_array_contents_kwargs(
    draw: st.DrawFn,
) -> KwargsDrawn[ListArrayContentsKwargs, ListArrayContentsDrawn]:
    """Strategy for options for `list_array_contents()` strategy."""

    @st.composite
    def _st_related_kwargs(draw: st.DrawFn) -> RelatedKwargs:
        """Strategy for the options that are drawn together."""
        min_length, max_length = draw(st_ak.misc.ranges(min_start=0, max_end=20))
        ret = RelatedKwargs()
        if min_length is not None:
            ret['min_length'] = min_length
        if max_length is not None:
            ret['max_length'] = max_length
        return ret

    related = draw(_st_related_kwargs())

    optional_independent = ListArrayContentsKwargs()
    drawn = ListArrayContentsDrawn(content=None)
    if draw(st.booleans()):
        content = draw(st_ak.constructors.contents())
        optional_independent['content'] = draw(
            st.sampled_from([content, st.just(content)])
        )
        drawn['content'] = content

    kwargs = ListArrayContentsKwargs(**related, **optional_independent)
    return KwargsDrawn(kwargs, drawn)


@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """Assert the results of `list_array_contents()`."""
    # Draw options
    kwargs, drawn = data.draw(list_array_contents_kwargs(), label='opts')

    # Call the test subject
    result = data.draw(st_ak.constructors.list_array_contents(**kwargs), label='result')

    assert isinstance(result, ListArray)

    # Assert the options were effective
    min_length = kwargs.get('min_length', DEFAULTS['min_length'])
    max_length = kwargs.get('max_length', DEFAULTS['max_length'])

    assert min_length <= len(result) <= sc(max_length)

    # Assert starts and stops have the same length
    starts = result.starts.data
    stops = result.stops.data
    assert len(starts) == len(stops)

    # Assert starts[i] <= stops[i] and stops do not exceed content length
    for i in range(len(starts)):
        assert starts[i] <= stops[i]
        assert stops[i] <= len(result.content)

    # Assert the options were effective
    if 'content' in kwargs:
        assert result.content is drawn['content']


@pytest.mark.parametrize('min_length', [1, 2, 10])
def test_draw_min_length(min_length: int) -> None:
    """Assert the length can reach `min_length`."""
    find(
        st_ak.constructors.list_array_contents(min_length=min_length),
        lambda c: len(c) == min_length,
        settings=FIND,
    )


@pytest.mark.parametrize('max_length', [1, 2, 10])
def test_draw_max_length(max_length: int) -> None:
    """Assert the length can reach `max_length`."""
    find(
        st_ak.constructors.list_array_contents(max_length=max_length),
        lambda c: len(c) == max_length,
        settings=FIND,
    )


@pytest.mark.parametrize('len_content', [0, 1, 2, 10])
def test_draw_default_max_length(len_content: int) -> None:
    """Assert that len(result) can reach len(content) by default."""
    content = NumpyArray(np.zeros(len_content))
    assert len(content) == len_content
    find(
        st_ak.constructors.list_array_contents(content),
        lambda c: len(c) == len(content),
        settings=FIND,
    )


def test_draw_unreachable() -> None:
    """Assert data can be unreachable."""
    content = NumpyArray(np.arange(10))
    find(
        st_ak.constructors.list_array_contents(content),
        lambda c: len(c) >= 1 and (c.starts[0] > 0 or c.stops[-1] < len(c.content)),
        settings=FIND,
    )


def test_shrink_no_unreachable() -> None:
    """Assert reachable data only is the simplest."""
    content = NumpyArray(np.arange(10))
    c = find(
        st_ak.constructors.list_array_contents(content),
        lambda c: len(c) >= 2,
        settings=FIND,
    )
    assert c.starts[0] == 0
    assert c.stops[-1] == len(c.content)


def test_shrink_content_len_zero() -> None:
    """Assert no sublists are the simplest for an empty content."""
    content = NumpyArray(np.array([], dtype=np.int64))
    c = find(
        st_ak.constructors.list_array_contents(content), lambda c: True, settings=FIND
    )
    assert len(c) == 0


def test_draw_from_contents() -> None:
    """Assert `contents()` can generate a `ListArray` as outermost."""
    find(
        st_ak.constructors.contents(), lambda c: isinstance(c, ListArray), settings=FIND
    )


def test_draw_from_contents_variable_length() -> None:
    """Assert that ListArray with variable-length sublists can be drawn."""

    def _has_variable_length(c: Content) -> bool:
        return any(
            isinstance(n, ListArray)
            and len(n) >= 2
            and len(set(len(n[i]) for i in range(len(n)))) > 1
            for n in iter_contents(c)
        )

    find(st_ak.constructors.contents(), _has_variable_length, settings=FIND_NO_SHRINK)


def test_draw_from_contents_empty_sublist() -> None:
    """Assert that ListArray with empty sublists can be drawn from `contents()`."""

    def _has_empty_sublist(c: Content) -> bool:
        return any(
            isinstance(n, ListArray) and any(len(n[i]) == 0 for i in range(len(n)))
            for n in iter_contents(c)
        )

    find(st_ak.constructors.contents(), _has_empty_sublist, settings=FIND_NO_SHRINK)
