from typing import Any, TypedDict, cast

import pytest
from hypothesis import find, given
from hypothesis import strategies as st

from awkward.contents import (
    BitMaskedArray,
    ByteMaskedArray,
    Content,
    UnmaskedArray,
)
from hypothesis_awkward import strategies as st_ak
from tests.find_settings import FIND
from tests.funcs import KwargsDrawn, assert_kwargs_match_signature

MASKED_TYPES = (ByteMaskedArray, BitMaskedArray, UnmaskedArray)


class MaskedContentsKwargs(TypedDict, total=False):
    """Options for `masked_contents()` strategy."""

    content: st.SearchStrategy[Content] | Content | None
    allow_byte_masked: bool
    allow_bit_masked: bool
    allow_unmasked: bool


DEFAULTS = MaskedContentsKwargs(
    content=None,
    allow_byte_masked=True,
    allow_bit_masked=True,
    allow_unmasked=True,
)

ALLOW_FLAGS = ('allow_byte_masked', 'allow_bit_masked', 'allow_unmasked')


class MaskedContentsDrawn(TypedDict):
    """Values drawn in advance for the options of `masked_contents()`."""

    content: Content | None


class RelatedKwargs(TypedDict, total=False):
    """Options drawn together (at least one masked type is allowed)."""

    allow_byte_masked: bool
    allow_bit_masked: bool
    allow_unmasked: bool


def test_kwargs_match_signature() -> None:
    """Assert the option declarations agree with `masked_contents()`."""
    assert_kwargs_match_signature(
        func=st_ak.contents.masked_contents,
        kwargs_cls=MaskedContentsKwargs,
        defaults=DEFAULTS,
        related_cls=RelatedKwargs,
    )


@st.composite
def masked_contents_kwargs(
    draw: st.DrawFn,
) -> KwargsDrawn[MaskedContentsKwargs, MaskedContentsDrawn]:
    """Strategy for options for `masked_contents()` strategy."""

    @st.composite
    def _st_related_kwargs(draw: st.DrawFn) -> RelatedKwargs:
        """Strategy for the options that are drawn together."""
        ret = dict[str, Any]()
        allowed = draw(st.sets(st.sampled_from(ALLOW_FLAGS), min_size=1))
        for flag in ALLOW_FLAGS:
            # A disallowed flag must be passed; an allowed one may be omitted.
            if flag not in allowed or draw(st.booleans()):
                ret[flag] = flag in allowed
        return cast(RelatedKwargs, ret)

    related = draw(_st_related_kwargs())

    optional_independent = MaskedContentsKwargs()
    drawn = MaskedContentsDrawn(content=None)
    if draw(st.booleans()):
        content = draw(
            st_ak.contents.contents(
                allow_union_root=False,
                allow_option_root=False,
                allow_indexed_root=False,
            )
        )
        optional_independent['content'] = draw(
            st.sampled_from([content, st.just(content)])
        )
        drawn['content'] = content

    kwargs = MaskedContentsKwargs(**related, **optional_independent)
    return KwargsDrawn(kwargs, drawn)


@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """Assert the results of `masked_contents()`."""
    # Draw options
    kwargs, drawn = data.draw(masked_contents_kwargs(), label='opts')

    allow_byte_masked = kwargs.get('allow_byte_masked', DEFAULTS['allow_byte_masked'])
    allow_bit_masked = kwargs.get('allow_bit_masked', DEFAULTS['allow_bit_masked'])
    allow_unmasked = kwargs.get('allow_unmasked', DEFAULTS['allow_unmasked'])

    # Call the test subject
    result = data.draw(st_ak.contents.masked_contents(**kwargs), label='result')

    assert isinstance(result, MASKED_TYPES)

    # Assert allow flags are respected
    if not allow_byte_masked:
        assert not isinstance(result, ByteMaskedArray)
    if not allow_bit_masked:
        assert not isinstance(result, BitMaskedArray)
    if not allow_unmasked:
        assert not isinstance(result, UnmaskedArray)

    # Assert content identity
    if 'content' in kwargs:
        assert result.content is drawn['content']


@pytest.mark.parametrize('cls', MASKED_TYPES)
def test_draw_masked_type(cls: type[Content]) -> None:
    """Assert the given masked type can be drawn."""
    find(st_ak.contents.masked_contents(), lambda c: isinstance(c, cls), settings=FIND)


def test_shrink_to_unmasked() -> None:
    """Assert `UnmaskedArray` is the simplest."""
    c = find(st_ak.contents.masked_contents(), lambda _: True, settings=FIND)
    assert isinstance(c, UnmaskedArray)


def test_shrink_to_byte_masked() -> None:
    """Assert `ByteMaskedArray` is the next simplest."""
    c = find(
        st_ak.contents.masked_contents(allow_unmasked=False),
        lambda _: True,
        settings=FIND,
    )
    assert isinstance(c, ByteMaskedArray)


def test_raise_all_disallowed() -> None:
    """Assert `ValueError` when every masked type is disallowed."""
    with pytest.raises(
        ValueError, match='at least one masked content type must be allowed'
    ):
        st_ak.contents.masked_contents(**{flag: False for flag in ALLOW_FLAGS})
