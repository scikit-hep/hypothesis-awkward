import math
from typing import Any, TypedDict, cast

import numpy as np
from hypothesis import Phase, find, given, settings
from hypothesis import strategies as st

from awkward.contents import BitMaskedArray, Content
from hypothesis_awkward import strategies as st_ak


class BitMaskedArrayContentsKwargs(TypedDict, total=False):
    """Options for `bit_masked_array_contents()` strategy."""

    content: st.SearchStrategy[Content] | Content


@st.composite
def bit_masked_array_contents_kwargs(
    draw: st.DrawFn,
    chain: st_ak.OptsChain[Any] | None = None,
) -> st_ak.OptsChain[BitMaskedArrayContentsKwargs]:
    """Strategy for options for `bit_masked_array_contents()` strategy."""
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

    return chain.extend(cast(BitMaskedArrayContentsKwargs, kwargs))


@settings(max_examples=200)
@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """Assert the results of `bit_masked_array_contents()`."""
    # Draw options
    opts = data.draw(bit_masked_array_contents_kwargs(), label='opts')
    opts.reset()

    # Call the test subject
    result = data.draw(
        st_ak.contents.bit_masked_array_contents(**opts.kwargs), label='result'
    )

    assert isinstance(result, BitMaskedArray)

    # Assert mask dtype is uint8
    assert result.mask.data.dtype == np.uint8

    # Assert logical length equals content length
    assert result.length == len(result.content)

    # Assert mask byte count is ceil(length / 8)
    assert len(result.mask) == math.ceil(result.length / 8)

    # Assert valid_when and lsb_order are bool
    assert isinstance(result.valid_when, bool)
    assert isinstance(result.lsb_order, bool)

    # Assert the options were effective
    content = opts.kwargs.get('content', None)
    match content:
        case Content():
            assert result.content is content
        case st_ak.RecordDraws():
            assert len(content.drawn) == 1
            assert result.content is content.drawn[0]


def test_draw_valid_when_true() -> None:
    """Assert that valid_when=True can be drawn."""
    find(
        st_ak.contents.bit_masked_array_contents(),
        lambda c: c.valid_when is True,
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_valid_when_false() -> None:
    """Assert that valid_when=False can be drawn."""
    find(
        st_ak.contents.bit_masked_array_contents(),
        lambda c: c.valid_when is False,
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_lsb_order_true() -> None:
    """Assert that lsb_order=True can be drawn."""
    find(
        st_ak.contents.bit_masked_array_contents(),
        lambda c: c.lsb_order is True,
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_lsb_order_false() -> None:
    """Assert that lsb_order=False can be drawn."""
    find(
        st_ak.contents.bit_masked_array_contents(),
        lambda c: c.lsb_order is False,
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_from_contents() -> None:
    """Assert `contents()` can generate a `BitMaskedArray` as outermost."""
    find(
        st_ak.contents.contents(),
        lambda c: isinstance(c, BitMaskedArray),
        settings=settings(max_examples=2000),
    )


def _is_valid(c: BitMaskedArray, j: int) -> bool:
    """Check whether element j is valid according to the bit mask."""
    byte = c.mask.data[j // 8]
    if c.lsb_order:
        bit = bool(byte & (1 << (j % 8)))
    else:
        bit = bool(byte & (128 >> (j % 8)))
    return bit == c.valid_when


def test_draw_with_none_values() -> None:
    """Assert that masked (None) entries can appear."""
    find(
        st_ak.contents.bit_masked_array_contents(),
        lambda c: c.length > 0 and any(not _is_valid(c, j) for j in range(c.length)),
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_all_valid() -> None:
    """Assert that all-valid arrays can be drawn."""
    find(
        st_ak.contents.bit_masked_array_contents(),
        lambda c: c.length > 0 and all(_is_valid(c, j) for j in range(c.length)),
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )
