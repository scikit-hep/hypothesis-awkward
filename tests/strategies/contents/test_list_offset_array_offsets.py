from typing import TypedDict, cast

from hypothesis import find, given
from hypothesis import strategies as st

from hypothesis_awkward import strategies as st_ak
from hypothesis_awkward._util import safe_compare as sc
from hypothesis_awkward.strategies.contents.list_offset_array import _st_offsets
from tests.find_settings import FIND
from tests.funcs import assert_kwargs_match_signature


class OffsetsKwargs(TypedDict, total=False):
    """Options for `_st_offsets()`."""

    min_length: int
    max_length: int | None
    allow_unreachable: bool


DEFAULTS = OffsetsKwargs(min_length=0, max_length=None, allow_unreachable=True)


class RelatedKwargs(TypedDict, total=False):
    """Options drawn together (`min_length <= max_length <= content_len`)."""

    min_length: int
    max_length: int | None


def test_kwargs_match_signature() -> None:
    """Assert the option declarations agree with `_st_offsets()`."""
    assert_kwargs_match_signature(
        func=_st_offsets,
        exclude={'content_len'},
        kwargs_cls=OffsetsKwargs,
        defaults=DEFAULTS,
        related_cls=RelatedKwargs,
    )


@st.composite
def offsets_kwargs(draw: st.DrawFn, content_len: int) -> OffsetsKwargs:
    """Strategy for options for `_st_offsets()`.

    The length bounds are drawn within `content_len`, the positional argument the
    caller passes to `_st_offsets()`.
    """

    @st.composite
    def _st_related_kwargs(draw: st.DrawFn) -> RelatedKwargs:
        """Strategy for the options that are drawn together."""
        min_length, max_length = draw(st_ak.ranges(min_start=0, max_end=content_len))
        ret = RelatedKwargs()
        if min_length is not None:
            ret['min_length'] = min_length
        if max_length is not None:
            ret['max_length'] = max_length
        return ret

    related = draw(_st_related_kwargs())

    optional_independent = draw(
        st.fixed_dictionaries(
            {},
            optional={
                'allow_unreachable': st.booleans(),
            },
        )
    )

    return cast(OffsetsKwargs, {**related, **optional_independent})


@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """Assert the results of `_st_offsets()`."""
    # Draw options
    content_len = data.draw(st.integers(min_value=0, max_value=50), label='content_len')
    kwargs = data.draw(offsets_kwargs(content_len=content_len), label='kwargs')

    # Call the test subject
    result = data.draw(_st_offsets(content_len, **kwargs), label='result')

    # Assert the options were effective
    min_length = kwargs.get('min_length', DEFAULTS['min_length'])
    max_length = kwargs.get('max_length', DEFAULTS['max_length'])
    allow_unreachable = kwargs.get('allow_unreachable', DEFAULTS['allow_unreachable'])

    # Offsets have at least one element
    assert len(result) >= 1

    # All offsets are within [0, content_len]
    assert all(0 <= o <= content_len for o in result)

    # Offsets are non-decreasing
    for i in range(len(result) - 1):
        assert result[i] <= result[i + 1]

    # Number of lists respects min_length / max_length
    n_lists = len(result) - 1
    assert min_length <= n_lists <= sc(max_length)

    if not allow_unreachable:
        if sc(max_length) >= 1:
            assert result[0] == 0
            assert result[-1] == content_len
        else:
            # unreachable occurs when max_length == 0
            assert max_length == 0
            assert result == [0]


def test_draw_min_length() -> None:
    """Assert that offsets with exactly min_length lists can be drawn."""
    find(_st_offsets(5, min_length=3), lambda o: len(o) - 1 == 3, settings=FIND)


def test_draw_min_length_content_len_zero() -> None:
    """Assert that min_length is respected when content is empty."""
    find(_st_offsets(0, min_length=3), lambda o: len(o) - 1 == 3, settings=FIND)


def test_draw_max_length() -> None:
    """Assert that offsets with exactly max_length lists can be drawn."""
    find(_st_offsets(5, max_length=10), lambda o: len(o) - 1 == 10, settings=FIND)


def test_draw_unreachable_head() -> None:
    """Assert that offsets with unreachable head data can be drawn."""
    find(_st_offsets(10), lambda o: len(o) >= 2 and o[0] > 0, settings=FIND)


def test_draw_unreachable_tail() -> None:
    """Assert that offsets with unreachable tail data can be drawn."""
    find(_st_offsets(10), lambda o: len(o) >= 2 and o[-1] < 10, settings=FIND)


def test_shrink_no_unreachable() -> None:
    """Assert that offsets shrink to no unreachable data."""
    offsets = find(_st_offsets(10), lambda o: len(o) >= 3, settings=FIND)
    assert offsets[0] == 0
    assert offsets[-1] == 10


def test_shrink_content_len_zero() -> None:
    """Assert that offsets shrink to one element (empty array) with no content."""
    offsets = find(_st_offsets(0), lambda o: len(o) < 3, settings=FIND)
    assert offsets == [0]
