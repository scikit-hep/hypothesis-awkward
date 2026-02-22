from typing import Any, Callable, TypedDict, cast

from hypothesis import Phase, find, given, settings
from hypothesis import strategies as st

import hypothesis_awkward.strategies as st_ak
from awkward.contents import Content
from hypothesis_awkward.strategies.misc.record import RecordCallDraws
from hypothesis_awkward.util import iter_leaf_contents
from hypothesis_awkward.util.safe import safe_compare as sc

DEFAULT_MAX_TOTAL_SIZE = 10
DEFAULT_MIN_SIZE = 0


class ContentListsKwargs(TypedDict, total=False):
    '''Options for `content_lists()` strategy.'''

    st_content: Callable[..., st.SearchStrategy[Content]]
    max_total_size: int
    min_size: int
    max_size: int | None


@st.composite
def content_lists_kwargs(
    draw: st.DrawFn,
    chain: st_ak.OptsChain[Any] | None = None,
) -> st_ak.OptsChain[ContentListsKwargs]:
    '''Strategy for options for `content_lists()` strategy.'''
    if chain is None:
        chain = st_ak.OptsChain({})
    st_content = chain.register_callable(st_ak.contents.contents)

    min_size, max_size = draw(st_ak.ranges(min_start=0, max_end=10, max_start=5))

    drawn = (
        ('min_size', min_size),
        ('max_size', max_size),
    )

    kwargs = draw(
        st.fixed_dictionaries(
            {k: st.just(v) for k, v in drawn if v is not None},
            optional={
                'st_content': st.just(st_content),
                'max_total_size': st.integers(min_value=0, max_value=50),
            },
        )
    )

    return chain.extend(cast(ContentListsKwargs, kwargs))


@settings(max_examples=200)
@given(data=st.data())
def test_content_lists(data: st.DataObject) -> None:
    '''Test that `content_lists()` respects all its options.'''
    opts = data.draw(content_lists_kwargs(), label='opts')
    opts.reset()

    max_total_size = opts.kwargs.get('max_total_size', DEFAULT_MAX_TOTAL_SIZE)
    min_size = opts.kwargs.get('min_size', DEFAULT_MIN_SIZE)
    max_size = opts.kwargs.get('max_size')

    result = data.draw(
        st_ak.contents.content_lists(**opts.kwargs),
        label='result',
    )

    assert isinstance(result, list)
    assert all(isinstance(c, Content) for c in result)
    assert sc(min_size) <= len(result) <= sc(max_size)
    assert _total_leaf_size(result) <= max_total_size

    match opts.kwargs.get('st_content'):
        case RecordCallDraws() as st_content:
            assert len(st_content.drawn) == len(result)
            assert all(d is r for d, r in zip(st_content.drawn, result))


def test_draw_min_size() -> None:
    '''Assert that a list with exactly min_size=2 elements can be drawn.'''
    find(
        st_ak.contents.content_lists(
            st_ak.contents.contents, max_total_size=50, min_size=2
        ),
        lambda cl: len(cl) == 2,
        settings=settings(phases=[Phase.generate]),
    )


def test_draw_max_size() -> None:
    '''Assert that max_size caps the number of contents.'''
    find(
        st_ak.contents.content_lists(max_total_size=50, max_size=3),
        lambda cl: len(cl) == 3,
        settings=settings(phases=[Phase.generate]),
    )


def test_draw_empty_list() -> None:
    '''Assert that an empty list can be drawn when min_size=0.'''
    find(
        st_ak.contents.content_lists(
            st_ak.contents.contents, max_total_size=50, min_size=0
        ),
        lambda cl: len(cl) == 0,
        settings=settings(phases=[Phase.generate]),
    )


def _total_leaf_size(contents: list[Content]) -> int:
    '''Total leaf elements across all contents.'''
    return sum(len(leaf) for c in contents for leaf in iter_leaf_contents(c))
