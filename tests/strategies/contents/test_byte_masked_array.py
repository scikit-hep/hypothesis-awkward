from typing import Any, TypedDict, cast

import numpy as np
from hypothesis import Phase, find, given, settings
from hypothesis import strategies as st

import hypothesis_awkward.strategies as st_ak
from awkward.contents import ByteMaskedArray, Content


class ByteMaskedArrayContentsKwargs(TypedDict, total=False):
    '''Options for `byte_masked_array_contents()` strategy.'''

    content: st.SearchStrategy[Content] | Content


@st.composite
def byte_masked_array_contents_kwargs(
    draw: st.DrawFn,
    chain: st_ak.OptsChain[Any] | None = None,
) -> st_ak.OptsChain[ByteMaskedArrayContentsKwargs]:
    '''Strategy for options for `byte_masked_array_contents()` strategy.'''
    if chain is None:
        chain = st_ak.OptsChain({})
    st_content = chain.register(
        st_ak.contents.contents(allow_union_root=False, allow_option_root=False)
    )

    kwargs = draw(
        st.fixed_dictionaries(
            {},
            optional={
                'content': st.one_of(
                    st_ak.contents.contents(
                        allow_union_root=False, allow_option_root=False
                    ),
                    st.just(st_content),
                ),
            },
        )
    )

    return chain.extend(cast(ByteMaskedArrayContentsKwargs, kwargs))


@settings(max_examples=200)
@given(data=st.data())
def test_byte_masked_array_contents(data: st.DataObject) -> None:
    '''Test that `byte_masked_array_contents()` respects all its options.'''
    # Draw options
    opts = data.draw(byte_masked_array_contents_kwargs(), label='opts')
    opts.reset()

    # Call the test subject
    result = data.draw(
        st_ak.contents.byte_masked_array_contents(**opts.kwargs), label='result'
    )

    assert isinstance(result, ByteMaskedArray)

    # Assert mask dtype is int8
    assert result.mask.data.dtype == np.int8

    # Assert mask length equals content length
    assert len(result.mask) == len(result.content)

    # Assert mask values are 0 or 1
    assert set(result.mask.data).issubset({0, 1})

    # Assert valid_when is bool
    assert isinstance(result.valid_when, bool)

    # Assert the options were effective
    content = opts.kwargs.get('content', None)
    match content:
        case Content():
            assert result.content is content
        case st_ak.RecordDraws():
            assert len(content.drawn) == 1
            assert result.content is content.drawn[0]


def test_draw_valid_when_true() -> None:
    '''Assert that valid_when=True can be drawn.'''
    find(
        st_ak.contents.byte_masked_array_contents(),
        lambda c: c.valid_when is True,
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_valid_when_false() -> None:
    '''Assert that valid_when=False can be drawn.'''
    find(
        st_ak.contents.byte_masked_array_contents(),
        lambda c: c.valid_when is False,
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_from_contents() -> None:
    '''Assert that ByteMaskedArray can be the root node from `contents()`.'''
    find(
        st_ak.contents.contents(),
        lambda c: isinstance(c, ByteMaskedArray),
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_with_none_values() -> None:
    '''Assert that masked (None) entries can appear.'''
    find(
        st_ak.contents.byte_masked_array_contents(),
        lambda c: (
            len(c) > 0 and any(c.mask.data[i] != c.valid_when for i in range(len(c)))
        ),
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_all_valid() -> None:
    '''Assert that all-valid arrays can be drawn.'''
    find(
        st_ak.contents.byte_masked_array_contents(),
        lambda c: (
            len(c) > 0 and all(c.mask.data[i] == c.valid_when for i in range(len(c)))
        ),
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )
