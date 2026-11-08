import math
from typing import TypedDict

import numpy as np
import pytest
from hypothesis import find, given
from hypothesis import strategies as st

from awkward.contents import BitMaskedArray, Content
from hypothesis_awkward import strategies as st_ak
from tests.find_settings import FIND
from tests.funcs import KwargsDrawn, assert_kwargs_match_signature


class BitMaskedArrayContentsKwargs(TypedDict, total=False):
    """Options for `bit_masked_array_contents()` strategy."""

    content: st.SearchStrategy[Content] | Content | None


DEFAULTS = BitMaskedArrayContentsKwargs(content=None)


class BitMaskedArrayContentsDrawn(TypedDict):
    """Values drawn in advance for the options of `bit_masked_array_contents()`."""

    content: Content | None


def test_kwargs_match_signature() -> None:
    """Assert the option declarations agree with `bit_masked_array_contents()`."""
    assert_kwargs_match_signature(
        func=st_ak.constructors.bit_masked_array_contents,
        kwargs_cls=BitMaskedArrayContentsKwargs,
        defaults=DEFAULTS,
    )


@st.composite
def bit_masked_array_contents_kwargs(
    draw: st.DrawFn,
) -> KwargsDrawn[BitMaskedArrayContentsKwargs, BitMaskedArrayContentsDrawn]:
    """Strategy for options for `bit_masked_array_contents()` strategy."""
    kwargs = BitMaskedArrayContentsKwargs()
    drawn = BitMaskedArrayContentsDrawn(content=None)
    if draw(st.booleans()):
        content = draw(
            st_ak.constructors.contents(
                allow_union_root=False,
                allow_option_root=False,
                allow_indexed_root=False,
            )
        )
        kwargs['content'] = draw(st.sampled_from([content, st.just(content)]))
        drawn['content'] = content
    return KwargsDrawn(kwargs, drawn)


@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """Assert the results of `bit_masked_array_contents()`."""
    # Draw options
    kwargs, drawn = data.draw(bit_masked_array_contents_kwargs(), label='opts')

    # Call the test subject
    result = data.draw(
        st_ak.constructors.bit_masked_array_contents(**kwargs), label='result'
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
    if 'content' in kwargs:
        assert result.content is drawn['content']


@pytest.mark.parametrize('valid_when', [True, False])
def test_draw_valid_when(valid_when: bool) -> None:
    """Assert the given `valid_when` can be drawn."""
    find(
        st_ak.constructors.bit_masked_array_contents(),
        lambda c: c.valid_when is valid_when,
        settings=FIND,
    )


@pytest.mark.parametrize('lsb_order', [True, False])
def test_draw_lsb_order(lsb_order: bool) -> None:
    """Assert the given `lsb_order` can be drawn."""
    find(
        st_ak.constructors.bit_masked_array_contents(),
        lambda c: c.lsb_order is lsb_order,
        settings=FIND,
    )


def test_draw_from_contents() -> None:
    """Assert `contents()` can generate a `BitMaskedArray` as outermost."""
    find(
        st_ak.constructors.contents(),
        lambda c: isinstance(c, BitMaskedArray),
        settings=FIND,
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
        st_ak.constructors.bit_masked_array_contents(),
        lambda c: c.length > 0 and any(not _is_valid(c, j) for j in range(c.length)),
        settings=FIND,
    )


def test_draw_all_valid() -> None:
    """Assert that all-valid arrays can be drawn."""
    find(
        st_ak.constructors.bit_masked_array_contents(),
        lambda c: c.length > 0 and all(_is_valid(c, j) for j in range(c.length)),
        settings=FIND,
    )
