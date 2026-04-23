from typing import Any, TypedDict, cast

import numpy as np
import pytest
from hypothesis import Phase, find, given, settings
from hypothesis import strategies as st

from awkward.contents import Content, NumpyArray, RegularArray
from hypothesis_awkward import strategies as st_ak
from hypothesis_awkward.util import safe_compare as sc


class RegularArrayContentsKwargs(TypedDict, total=False):
    """Options for `regular_array_contents()` strategy."""

    content: st.SearchStrategy[Content] | Content
    max_size: int | None
    max_zeros_length: int | None
    max_length: int | None


@st.composite
def regular_array_contents_kwargs(
    draw: st.DrawFn,
    chain: st_ak.OptsChain[Any] | None = None,
) -> st_ak.OptsChain[RegularArrayContentsKwargs]:
    """Strategy for options for `regular_array_contents()` strategy."""
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
                'max_size': st.integers(min_value=0, max_value=50),
                'max_zeros_length': st.integers(min_value=0, max_value=50),
                'max_length': st.integers(min_value=0, max_value=50),
            },
        )
    )

    return chain.extend(cast(RegularArrayContentsKwargs, kwargs))


@settings(max_examples=200)
@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """Assert the results of `regular_array_contents()`."""
    # Draw options
    opts = data.draw(regular_array_contents_kwargs(), label='opts')
    opts.reset()

    # Call the test subject
    result = data.draw(
        st_ak.contents.regular_array_contents(**opts.kwargs), label='result'
    )

    assert isinstance(result, RegularArray)

    # Assert the options were effective
    max_size = opts.kwargs.get('max_size')
    assert result.size <= sc(max_size)

    # Assert zeros_length is within bounds
    max_zeros_length = opts.kwargs.get('max_zeros_length')
    if result.size == 0:
        assert len(result) <= sc(max_zeros_length)

    # Assert length is within bounds
    max_length = opts.kwargs.get('max_length')
    assert len(result) <= sc(max_length)

    # Assert size divides content length when size > 0
    # TODO: Re-enable when allow_unreachable option is added to regular_array_contents().
    # if result.size > 0:
    #     assert len(result.content) % result.size == 0
    #     assert len(result) == len(result.content) // result.size

    # Assert content
    content = opts.kwargs.get('content', None)
    match content:
        case Content():
            assert result.content is content
        case st_ak.RecordDraws():
            assert len(content.drawn) == 1
            assert result.content is content.drawn[0]


def test_draw_size_zero() -> None:
    """Assert the size can be zero."""
    find(st_ak.contents.regular_array_contents(), lambda c: c.size == 0)


@pytest.mark.parametrize('max_size', [0, 1, 2, 50])
def test_draw_max_size(max_size: int) -> None:
    """Assert the size can reach `max_size`."""
    find(
        st_ak.contents.regular_array_contents(max_size=max_size),
        lambda c: c.size == max_size,
    )


@pytest.mark.parametrize('max_length', [1, 2, 8])
def test_draw_max_length(max_length: int) -> None:
    """Assert the length can reach `max_length`."""
    find(
        st_ak.contents.regular_array_contents(max_length=max_length),
        lambda c: c.size > 0 and len(c) == max_length,
    )


@pytest.mark.parametrize('max_zeros_length', [0, 1, 2, 50])
def test_draw_max_zeros_length(max_zeros_length: int) -> None:
    """Assert the length can reach `max_zeros_length` when size is zero."""
    find(
        st_ak.contents.regular_array_contents(max_zeros_length=max_zeros_length),
        lambda c: c.size == 0 and len(c) == max_zeros_length,
    )


@pytest.mark.parametrize('max_length', [0, 1, 2, 10])
def test_draw_default_max_size(max_length: int) -> None:
    """Assert the size can reach `len(content)` when `max_size` is not specified."""
    content = st_ak.contents.contents(max_length=max_length)
    find(
        st_ak.contents.regular_array_contents(content),
        lambda c: c.size == max_length,
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


@pytest.mark.parametrize('max_length', [0, 1, 2, 10])
def test_draw_default_max_zeros_length(max_length: int) -> None:
    """Assert the length can reach `len(content)` when the size is zero."""
    content = st_ak.contents.contents(max_length=max_length)
    find(
        st_ak.contents.regular_array_contents(content, max_size=0),
        lambda c: c.size == 0 and len(c) == max_length,
    )


def test_draw_unreachable() -> None:
    """Assert data can be unreachable."""
    find(
        st_ak.contents.regular_array_contents(),
        lambda c: c.size > 0 and len(c.content) % c.size != 0,
    )


@pytest.mark.parametrize('n', [6, 8, 12, 15, 16])
def test_shrink_no_unreachable(n: int) -> None:
    """Assert reachable data only is the simplest."""
    # E.g., n=15
    # Content of length 15: divisors in (1, 15) are [5, 3],
    # non-divisors in (1, 15) are [14, 13, ..., 6, 4, 2].
    # Shrink should pick 5 (largest divisor < 15), not 14 (largest non-divisor).
    content = NumpyArray(np.arange(n))
    c = find(
        st_ak.contents.regular_array_contents(content),
        lambda c: 1 < c.size < len(c.content),
    )
    assert len(c) == min(n // s for s in range(2, n) if n % s == 0)
    assert len(c.content) == c.size * len(c)


def test_draw_from_contents() -> None:
    """Assert `contents()` can generate a `RegularArray` as outermost."""
    find(st_ak.contents.contents(), lambda c: isinstance(c, RegularArray))
