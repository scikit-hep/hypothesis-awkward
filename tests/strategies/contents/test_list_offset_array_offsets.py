from typing import TypedDict, cast

from hypothesis import find, given, settings
from hypothesis import strategies as st

from hypothesis_awkward.strategies.contents.list_offset_array import _st_offsets
from hypothesis_awkward.util import safe_compare as sc


class OffsetsKwargs(TypedDict, total=False):
    """Options for `_st_offsets()`."""

    content_len: int
    max_length: int | None
    allow_unreachable: bool


@st.composite
def offsets_kwargs(draw: st.DrawFn) -> OffsetsKwargs:
    """Strategy for options for `_st_offsets()`."""
    content_len = draw(st.integers(min_value=0, max_value=50))

    drawn = (('content_len', content_len),)

    kwargs = draw(
        st.fixed_dictionaries(
            {k: st.just(v) for k, v in drawn if v is not None},
            optional={
                'max_length': st.integers(min_value=0, max_value=content_len),
                'allow_unreachable': st.booleans(),
            },
        )
    )

    return cast(OffsetsKwargs, kwargs)


@settings(max_examples=500)
@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """Assert the results of `_st_offsets()`."""
    # Draw options
    kwargs = data.draw(offsets_kwargs(), label='kwargs')

    # Call the test subject
    result = data.draw(_st_offsets(**kwargs), label='result')

    # Assert the options were effective
    content_len = kwargs['content_len']
    max_length = kwargs.get('max_length')
    allow_unreachable = kwargs.get('allow_unreachable', True)

    # Offsets have at least one element
    assert len(result) >= 1

    # All offsets are within [0, content_len]
    assert all(0 <= o <= content_len for o in result)

    # Offsets are non-decreasing
    for i in range(len(result) - 1):
        assert result[i] <= result[i + 1]

    # Number of lists respects max_length
    n_lists = len(result) - 1
    assert n_lists <= sc(max_length)

    if not allow_unreachable:
        if sc(max_length) >= 1:
            assert result[0] == 0
            assert result[-1] == content_len
        else:
            # unreachable occurs when max_length == 0
            assert max_length == 0
            assert result == [0]


def test_draw_max_length() -> None:
    """Assert that offsets with exactly max_length lists can be drawn."""
    find(_st_offsets(5, max_length=10), lambda o: len(o) - 1 == 10)


def test_draw_unreachable_head() -> None:
    """Assert that offsets with unreachable head data can be drawn."""
    find(_st_offsets(10), lambda o: len(o) >= 2 and o[0] > 0)


def test_draw_unreachable_tail() -> None:
    """Assert that offsets with unreachable tail data can be drawn."""
    find(_st_offsets(10), lambda o: len(o) >= 2 and o[-1] < 10)


def test_shrink_no_unreachable() -> None:
    """Assert that offsets shrink to no unreachable data."""
    offsets = find(_st_offsets(10), lambda o: len(o) >= 3)
    assert offsets[0] == 0
    assert offsets[-1] == 10


def test_shrink_content_len_zero() -> None:
    """Assert that offsets shrink to one element (empty array) with no content."""
    offsets = find(_st_offsets(0), lambda o: len(o) < 3)
    assert offsets == [0]
