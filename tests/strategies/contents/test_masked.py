from contextlib import ExitStack
from typing import Any, TypedDict, cast

import pytest
from hypothesis import find, given, settings
from hypothesis import strategies as st

from awkward.contents import (
    BitMaskedArray,
    ByteMaskedArray,
    Content,
    UnmaskedArray,
)
from hypothesis_awkward import strategies as st_ak

MASKED_TYPES = (ByteMaskedArray, BitMaskedArray, UnmaskedArray)


class MaskedContentsKwargs(TypedDict, total=False):
    """Options for `masked_contents()` strategy."""

    content: st.SearchStrategy[Content] | Content
    allow_byte_masked: bool
    allow_bit_masked: bool
    allow_unmasked: bool


@st.composite
def masked_contents_kwargs(
    draw: st.DrawFn,
    chain: st_ak.OptsChain[Any] | None = None,
) -> st_ak.OptsChain[MaskedContentsKwargs]:
    """Strategy for options for `masked_contents()` strategy."""
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
                'allow_byte_masked': st.booleans(),
                'allow_bit_masked': st.booleans(),
                'allow_unmasked': st.booleans(),
            },
        )
    )

    return chain.extend(cast(MaskedContentsKwargs, kwargs))


@settings(max_examples=200)
@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """Assert the results of `masked_contents()`."""
    # Draw options
    opts = data.draw(masked_contents_kwargs(), label='opts')
    opts.reset()

    allow_byte_masked = opts.kwargs.get('allow_byte_masked', True)
    allow_bit_masked = opts.kwargs.get('allow_bit_masked', True)
    allow_unmasked = opts.kwargs.get('allow_unmasked', True)

    # Call the test subject
    expect_raised = False
    with ExitStack() as stack:
        if not any((allow_byte_masked, allow_bit_masked, allow_unmasked)):
            expect_raised = True
            stack.enter_context(
                pytest.raises(
                    ValueError,
                    match='at least one masked content type must be allowed',
                )
            )
        result = data.draw(
            st_ak.contents.masked_contents(**opts.kwargs), label='result'
        )

    if expect_raised:
        return

    assert isinstance(result, MASKED_TYPES)

    # Assert allow flags are respected
    if not allow_byte_masked:
        assert not isinstance(result, ByteMaskedArray)
    if not allow_bit_masked:
        assert not isinstance(result, BitMaskedArray)
    if not allow_unmasked:
        assert not isinstance(result, UnmaskedArray)

    # Assert content identity
    content = opts.kwargs.get('content', None)
    match content:
        case Content():
            assert result.content is content
        case st_ak.RecordDraws():
            assert len(content.drawn) == 1
            assert result.content is content.drawn[0]


@pytest.mark.parametrize('cls', MASKED_TYPES)
def test_draw_masked_type(cls: type[Content]) -> None:
    """Assert the given masked type can be drawn."""
    find(st_ak.contents.masked_contents(), lambda c: isinstance(c, cls))


def test_shrink_to_unmasked() -> None:
    """Assert `UnmaskedArray` is the simplest."""
    c = find(
        st_ak.contents.masked_contents(),
        lambda _: True,
        settings=settings(database=None),
    )
    assert isinstance(c, UnmaskedArray)


def test_shrink_to_byte_masked() -> None:
    """Assert `ByteMaskedArray` is the next simplest."""
    c = find(
        st_ak.contents.masked_contents(allow_unmasked=False),
        lambda _: True,
        settings=settings(database=None),
    )
    assert isinstance(c, ByteMaskedArray)
