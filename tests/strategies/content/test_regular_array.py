from typing import TypedDict, cast

import numpy as np
import pytest
from hypothesis import find, given
from hypothesis import strategies as st

from awkward.contents import Content, NumpyArray, RegularArray
from hypothesis_awkward import strategies as st_ak
from hypothesis_awkward._util import safe_compare as sc
from tests.find_settings import FIND, FIND_NO_SHRINK, FIND_RARE
from tests.funcs import KwargsDrawn, assert_kwargs_match_signature


class RegularArrayContentsKwargs(TypedDict, total=False):
    """Options for `regular_array_contents()` strategy."""

    content: st.SearchStrategy[Content] | Content | None
    max_size: int | None
    max_zeros_length: int | None
    min_length: int
    max_length: int | None


DEFAULTS = RegularArrayContentsKwargs(
    content=None,
    max_size=None,
    max_zeros_length=None,
    min_length=0,
    max_length=None,
)


class RegularArrayContentsDrawn(TypedDict, total=False):
    """Values drawn in advance for the options of `regular_array_contents()`."""

    content: Content


class RelatedKwargs(TypedDict, total=False):
    """Options drawn together (`min_length` <= `max_length`, `max_zeros_length`)."""

    max_zeros_length: int | None
    min_length: int
    max_length: int | None


def test_kwargs_match_signature() -> None:
    """Assert the option declarations agree with `regular_array_contents()`."""
    assert_kwargs_match_signature(
        func=st_ak.regular_array_contents,
        kwargs_cls=RegularArrayContentsKwargs,
        defaults=DEFAULTS,
        related_cls=RelatedKwargs,
    )


@st.composite
def regular_array_contents_kwargs(
    draw: st.DrawFn,
) -> KwargsDrawn[RegularArrayContentsKwargs, RegularArrayContentsDrawn]:
    """Strategy for options for `regular_array_contents()` strategy."""

    @st.composite
    def _st_related_kwargs(draw: st.DrawFn) -> RelatedKwargs:
        """Strategy for the options that are drawn together."""
        min_length, max_length = draw(st_ak.misc.ranges(min_start=0, max_end=20))
        max_zeros_length = draw(
            st_ak.misc.none_or(st.integers(min_value=min_length or 0, max_value=50))
        )
        ret = RelatedKwargs()
        if max_zeros_length is not None:
            ret['max_zeros_length'] = max_zeros_length
        if min_length is not None:
            ret['min_length'] = min_length
        if max_length is not None:
            ret['max_length'] = max_length
        return ret

    @st.composite
    def _st_drawn(draw: st.DrawFn) -> RegularArrayContentsDrawn:
        """Strategy for the values drawn in advance for the options."""
        ret = RegularArrayContentsDrawn()
        if draw(st.booleans()):
            ret['content'] = draw(st_ak.contents())
        return ret

    @st.composite
    def _st_optional_independent(
        draw: st.DrawFn, drawn: RegularArrayContentsDrawn
    ) -> RegularArrayContentsKwargs:
        """Strategy for the options that are drawn independently, if at all."""
        ret = cast(
            RegularArrayContentsKwargs,
            draw(
                st.fixed_dictionaries(
                    {}, optional={'max_size': st.integers(min_value=0, max_value=50)}
                )
            ),
        )
        if 'content' in drawn:
            content = drawn['content']
            ret['content'] = draw(st.sampled_from([content, st.just(content)]))
        return ret

    related = draw(_st_related_kwargs())
    drawn = draw(_st_drawn())
    optional_independent = draw(_st_optional_independent(drawn))

    kwargs = RegularArrayContentsKwargs(**related, **optional_independent)
    return KwargsDrawn(kwargs, drawn)


@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """Assert the results of `regular_array_contents()`."""
    # Draw options
    kwargs, drawn = data.draw(regular_array_contents_kwargs(), label='opts')

    # Call the test subject
    result = data.draw(st_ak.regular_array_contents(**kwargs), label='result')

    assert isinstance(result, RegularArray)

    # Assert the options were effective
    max_size = kwargs.get('max_size', DEFAULTS['max_size'])
    max_zeros_length = kwargs.get('max_zeros_length', DEFAULTS['max_zeros_length'])
    min_length = kwargs.get('min_length', DEFAULTS['min_length'])
    max_length = kwargs.get('max_length', DEFAULTS['max_length'])

    assert result.size <= sc(max_size)

    if result.size == 0:
        assert len(result) <= sc(max_zeros_length)

    assert min_length <= len(result) <= sc(max_length)

    # Assert size divides content length when size > 0
    # TODO: Re-enable when allow_unreachable option is added to regular_array_contents().
    # if result.size > 0:
    #     assert len(result.content) % result.size == 0
    #     assert len(result) == len(result.content) // result.size

    # Assert content
    if 'content' in kwargs:
        assert result.content is drawn['content']


def test_draw_size_zero() -> None:
    """Assert the size can be zero."""
    find(st_ak.regular_array_contents(), lambda c: c.size == 0, settings=FIND)


@pytest.mark.parametrize('max_size', [0, 1, 2, 50])
def test_draw_max_size(max_size: int) -> None:
    """Assert the size can reach `max_size`."""
    find(
        st_ak.regular_array_contents(max_size=max_size),
        lambda c: c.size == max_size,
        settings=FIND,
    )


@pytest.mark.parametrize('min_length', [1, 2, 8])
def test_draw_min_length(min_length: int) -> None:
    """Assert the length can reach `min_length` exactly."""
    find(
        st_ak.regular_array_contents(min_length=min_length),
        lambda c: c.size > 0 and len(c) == min_length,
        settings=FIND,
    )


@pytest.mark.parametrize('max_length', [1, 2, 8])
def test_draw_max_length(max_length: int) -> None:
    """Assert the length can reach `max_length`."""
    find(
        st_ak.regular_array_contents(max_length=max_length),
        lambda c: c.size > 0 and len(c) == max_length,
        settings=FIND_RARE,
    )


@pytest.mark.parametrize('max_zeros_length', [0, 1, 2, 50])
def test_draw_max_zeros_length(max_zeros_length: int) -> None:
    """Assert the length can reach `max_zeros_length` when size is zero."""
    find(
        st_ak.regular_array_contents(max_zeros_length=max_zeros_length),
        lambda c: c.size == 0 and len(c) == max_zeros_length,
        settings=FIND,
    )


@pytest.mark.parametrize('max_length', [0, 1, 2, 10])
def test_draw_default_max_size(max_length: int) -> None:
    """Assert the size can reach `len(content)` when `max_size` is not specified."""
    content = st_ak.contents(max_length=max_length)
    find(
        st_ak.regular_array_contents(content),
        lambda c: c.size == max_length,
        settings=FIND_NO_SHRINK,
    )


@pytest.mark.parametrize('max_length', [0, 1, 2, 10])
def test_draw_default_max_zeros_length(max_length: int) -> None:
    """Assert the length can reach `len(content)` when the size is zero."""
    content = st_ak.contents(max_length=max_length)
    find(
        st_ak.regular_array_contents(content, max_size=0),
        lambda c: c.size == 0 and len(c) == max_length,
        settings=FIND,
    )


def test_draw_unreachable() -> None:
    """Assert data can be unreachable."""
    find(
        st_ak.regular_array_contents(),
        lambda c: c.size > 0 and len(c.content) % c.size != 0,
        settings=FIND,
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
        st_ak.regular_array_contents(content),
        lambda c: 1 < c.size < len(c.content),
        settings=FIND,
    )
    assert len(c) == min(n // s for s in range(2, n) if n % s == 0)
    assert len(c.content) == c.size * len(c)


def test_draw_from_contents() -> None:
    """Assert `contents()` can generate a `RegularArray` as outermost."""
    find(st_ak.contents(), lambda c: isinstance(c, RegularArray), settings=FIND)
