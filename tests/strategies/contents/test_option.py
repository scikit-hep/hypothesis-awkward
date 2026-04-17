from typing import Any, TypedDict, cast

import pytest
from hypothesis import Phase, find, given, settings
from hypothesis import strategies as st

from awkward.contents import (
    BitMaskedArray,
    ByteMaskedArray,
    Content,
    IndexedOptionArray,
    UnmaskedArray,
)
from hypothesis_awkward import strategies as st_ak

OPTION_TYPES = (IndexedOptionArray, ByteMaskedArray, BitMaskedArray, UnmaskedArray)


class OptionContentsKwargs(TypedDict, total=False):
    """Options for `option_contents()` strategy."""

    content: st.SearchStrategy[Content] | Content
    max_size: int | None
    allow_indexed_option: bool
    allow_byte_masked: bool
    allow_bit_masked: bool
    allow_unmasked: bool


@st.composite
def option_contents_kwargs(
    draw: st.DrawFn,
    chain: st_ak.OptsChain[Any] | None = None,
) -> st_ak.OptsChain[OptionContentsKwargs]:
    """Strategy for options for `option_contents()` strategy."""
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
                'max_size': st.integers(min_value=0, max_value=50),
                'allow_indexed_option': st.booleans(),
                'allow_byte_masked': st.booleans(),
                'allow_bit_masked': st.booleans(),
                'allow_unmasked': st.booleans(),
            },
        )
    )

    return chain.extend(cast(OptionContentsKwargs, kwargs))


@settings(max_examples=200)
@given(data=st.data())
def test_option_contents(data: st.DataObject) -> None:
    """Test that `option_contents()` respects all its options."""
    # Draw options
    opts = data.draw(option_contents_kwargs(), label='opts')
    opts.reset()

    allow_indexed_option = opts.kwargs.get('allow_indexed_option', True)
    allow_byte_masked = opts.kwargs.get('allow_byte_masked', True)
    allow_bit_masked = opts.kwargs.get('allow_bit_masked', True)
    allow_unmasked = opts.kwargs.get('allow_unmasked', True)

    # If all are False, expect ValueError
    if not any(
        (allow_indexed_option, allow_byte_masked, allow_bit_masked, allow_unmasked)
    ):
        with pytest.raises(
            ValueError, match='at least one option content type must be allowed'
        ):
            st_ak.contents.option_contents(**opts.kwargs)
        return

    # Call the test subject
    result = data.draw(st_ak.contents.option_contents(**opts.kwargs), label='result')

    assert isinstance(result, OPTION_TYPES)

    # Assert allow flags are respected
    if not allow_indexed_option:
        assert not isinstance(result, IndexedOptionArray)
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


def test_draw_indexed_option() -> None:
    """Assert that IndexedOptionArray can be drawn."""
    find(
        st_ak.contents.option_contents(),
        lambda c: isinstance(c, IndexedOptionArray),
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_byte_masked() -> None:
    """Assert that ByteMaskedArray can be drawn."""
    find(
        st_ak.contents.option_contents(),
        lambda c: isinstance(c, ByteMaskedArray),
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_bit_masked() -> None:
    """Assert that BitMaskedArray can be drawn."""
    find(
        st_ak.contents.option_contents(),
        lambda c: isinstance(c, BitMaskedArray),
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_unmasked() -> None:
    """Assert that UnmaskedArray can be drawn."""
    find(
        st_ak.contents.option_contents(),
        lambda c: isinstance(c, UnmaskedArray),
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )
