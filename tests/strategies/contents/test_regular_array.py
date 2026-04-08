from typing import Any, TypedDict, cast

import numpy as np
from hypothesis import Phase, find, given, settings
from hypothesis import strategies as st

import hypothesis_awkward.strategies as st_ak
from awkward.contents import Content, NumpyArray, RegularArray
from hypothesis_awkward.util import iter_contents
from hypothesis_awkward.util.safe import safe_compare as sc


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
                'max_size': st_ak.none_or(st.integers(min_value=0, max_value=50)),
                'max_zeros_length': st_ak.none_or(
                    st.integers(min_value=0, max_value=50)
                ),
                'max_length': st.integers(min_value=0, max_value=50),
            },
        )
    )

    return chain.extend(cast(RegularArrayContentsKwargs, kwargs))


@settings(max_examples=200)
@given(data=st.data())
def test_regular_array_contents(data: st.DataObject) -> None:
    """Test that `regular_array_contents()` respects all its options."""
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


def test_draw_max_size() -> None:
    """Assert that RegularArray with exactly max_size can be drawn."""
    max_size = 10
    find(
        st_ak.contents.regular_array_contents(max_size=max_size),
        lambda c: c.size == max_size,
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_max_zeros_length() -> None:
    """Assert that zeros_length up to max_zeros_length can be drawn."""
    max_zeros_length = 20
    find(
        st_ak.contents.regular_array_contents(max_zeros_length=max_zeros_length),
        lambda c: c.size == 0 and len(c) == max_zeros_length,
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_max_length() -> None:
    """Assert that max_length constrains the RegularArray length."""
    max_length = 10
    content = st_ak.contents.numpy_array_contents(min_size=10, max_size=30)
    find(
        st_ak.contents.regular_array_contents(content, max_length=max_length),
        lambda c: c.size > 0 and len(c) == max_length,
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_default_max_size() -> None:
    """Assert that size can reach len(content) when max_size is not specified."""
    content = st_ak.contents.numpy_array_contents(max_size=12)
    find(
        st_ak.contents.regular_array_contents(content),
        lambda c: c.size == 12,
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_default_max_zeros_length() -> None:
    """Assert that zeros_length can reach len(content) when defaults are used."""
    content = st_ak.contents.numpy_array_contents(max_size=10)
    find(
        st_ak.contents.regular_array_contents(content, max_size=0),
        lambda c: c.size == 0 and len(c) == 10,
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_unreachable() -> None:
    """Assert that RegularArray with unreachable data can be drawn."""
    content = NumpyArray(np.arange(15))
    find(
        st_ak.contents.regular_array_contents(content),
        lambda c: c.size > 0 and len(c.content) % c.size != 0,
        settings=settings(phases=[Phase.generate]),
    )


def test_shrink_no_unreachable() -> None:
    """Assert that RegularArray shrinks to no unreachable data."""
    # Content of length 15: divisors in (1, 15) are [5, 3],
    # non-divisors in (1, 15) are [14, 13, ..., 6, 4, 2].
    # Shrink should pick 5 (largest divisor < 15), not 14 (largest non-divisor).
    content = NumpyArray(np.arange(15))
    c = find(
        st_ak.contents.regular_array_contents(content),
        lambda c: 1 < c.size < 15,
    )
    assert c.size == 5
    assert len(c.content) % c.size == 0


def test_draw_from_contents_size_zero() -> None:
    """Assert that RegularArray with size=0 can be drawn from `contents()`."""

    def _has_regular_size_zero(c: Content) -> bool:
        return any(
            isinstance(n, RegularArray) and n.size == 0 for n in iter_contents(c)
        )

    find(
        st_ak.contents.contents(),
        _has_regular_size_zero,
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )
