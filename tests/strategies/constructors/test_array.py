from typing import Any
from unittest.mock import Mock, patch

from hypothesis import given, settings
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak
from awkward.contents import Content
from tests.strategies.contents.test_content import ContentsKwargs, contents_kwargs


class ArraysKwargs(ContentsKwargs, total=False):
    '''Options for `arrays()` strategy.'''


DEFAULTS = ArraysKwargs(
    dtypes=None,
    max_size=10,
    allow_nan=False,
    allow_numpy=True,
    allow_empty=True,
    allow_string=True,
    allow_bytestring=True,
    allow_regular=True,
    allow_list_offset=True,
    allow_list=True,
    max_depth=5,
)


@st.composite
def arrays_kwargs(
    draw: st.DrawFn,
    chain: st_ak.OptsChain[Any] | None = None,
) -> st_ak.OptsChain[ArraysKwargs]:
    '''Strategy for options for `arrays()` strategy.'''
    return draw(contents_kwargs(chain=chain))


@settings(max_examples=200)
@given(data=st.data())
def test_arrays(data: st.DataObject) -> None:
    '''Test that `arrays()` forwards kwargs to `contents()` and wraps in `ak.Array`.'''
    from hypothesis_awkward.strategies.constructors import array_ as array_module

    opts = data.draw(arrays_kwargs(), label='opts')
    opts.reset()
    kwargs = opts.kwargs

    spy = ContentsSpy()
    mock_st_ak = Mock(wraps=st_ak)
    mock_st_ak.contents.contents = Mock(side_effect=spy.wrap_contents)

    with patch.object(array_module, 'st_ak', mock_st_ak):
        try:
            a = data.draw(st_ak.constructors.arrays(**kwargs), label='a')
        except Exception as e:
            assert spy.raised_exc is not None
            assert e is spy.raised_exc
        else:
            assert isinstance(a, ak.Array)
            assert a.layout is spy.drawn_layout

    mock_st_ak.contents.contents.assert_called_once_with(**{**DEFAULTS, **kwargs})


class ContentsSpy:
    '''Wraps `contents()` to capture the drawn layout and any raised exception.'''

    def __init__(self) -> None:
        self.drawn_layout: Content | None = None
        self.raised_exc: Exception | None = None

    def wrap_contents(self, *a: Any, **kw: Any) -> st.SearchStrategy[Content]:
        strategy = st_ak.contents.contents(*a, **kw)
        spy = self

        @st.composite
        def _draw(draw_inner: st.DrawFn) -> Content:
            spy.drawn_layout = None
            spy.raised_exc = None
            try:
                spy.drawn_layout = draw_inner(strategy)
            except Exception as e:
                spy.raised_exc = e
                raise
            return spy.drawn_layout

        return _draw()
