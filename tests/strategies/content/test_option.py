from typing import Any, TypedDict, cast

import pytest
from hypothesis import find, given
from hypothesis import strategies as st

from awkward.contents import (
    BitMaskedArray,
    ByteMaskedArray,
    Content,
    IndexedOptionArray,
    UnmaskedArray,
)
from hypothesis_awkward import strategies as st_ak
from hypothesis_awkward._util import safe_compare as sc
from tests.find_settings import FIND
from tests.funcs import KwargsDrawn, assert_kwargs_match_signature

OPTION_TYPES = (IndexedOptionArray, ByteMaskedArray, BitMaskedArray, UnmaskedArray)


class OptionContentsKwargs(TypedDict, total=False):
    """Options for `option_contents()` strategy."""

    content: st.SearchStrategy[Content] | Content | None
    min_size: int
    max_size: int | None
    allow_indexed_option: bool
    allow_byte_masked: bool
    allow_bit_masked: bool
    allow_unmasked: bool


DEFAULTS = OptionContentsKwargs(
    content=None,
    min_size=0,
    max_size=None,
    allow_indexed_option=True,
    allow_byte_masked=True,
    allow_bit_masked=True,
    allow_unmasked=True,
)

ALLOW_FLAGS = (
    'allow_indexed_option',
    'allow_byte_masked',
    'allow_bit_masked',
    'allow_unmasked',
)


class OptionContentsDrawn(TypedDict):
    """Values drawn in advance for the options of `option_contents()`."""

    content: Content | None


class RelatedKwargs(TypedDict, total=False):
    """Options drawn together.

    `min_size <= max_size`, and at least one option type is allowed.
    """

    min_size: int
    max_size: int | None
    allow_indexed_option: bool
    allow_byte_masked: bool
    allow_bit_masked: bool
    allow_unmasked: bool


def test_kwargs_match_signature() -> None:
    """Assert the option declarations agree with `option_contents()`."""
    assert_kwargs_match_signature(
        func=st_ak.content.option_contents,
        kwargs_cls=OptionContentsKwargs,
        defaults=DEFAULTS,
        related_cls=RelatedKwargs,
    )


@st.composite
def option_contents_kwargs(
    draw: st.DrawFn,
) -> KwargsDrawn[OptionContentsKwargs, OptionContentsDrawn]:
    """Strategy for options for `option_contents()` strategy."""

    @st.composite
    def _st_related_kwargs(draw: st.DrawFn) -> RelatedKwargs:
        """Strategy for the options that are drawn together."""
        min_size, max_size = draw(st_ak.misc.ranges(min_start=0, max_end=50))
        ret = dict[str, Any]()
        if min_size is not None:
            ret['min_size'] = min_size
        if max_size is not None:
            ret['max_size'] = max_size
        allowed = draw(st.sets(st.sampled_from(ALLOW_FLAGS), min_size=1))
        for flag in ALLOW_FLAGS:
            # A disallowed flag must be passed; an allowed one may be omitted.
            if flag not in allowed or draw(st.booleans()):
                ret[flag] = flag in allowed
        return cast(RelatedKwargs, ret)

    related = draw(_st_related_kwargs())

    optional_independent = OptionContentsKwargs()
    drawn = OptionContentsDrawn(content=None)
    if draw(st.booleans()):
        content = draw(
            st_ak.contents(
                allow_union_root=False,
                allow_option_root=False,
                allow_indexed_root=False,
            )
        )
        optional_independent['content'] = draw(
            st.sampled_from([content, st.just(content)])
        )
        drawn['content'] = content

    kwargs = OptionContentsKwargs(**related, **optional_independent)
    return KwargsDrawn(kwargs, drawn)


@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """Assert the results of `option_contents()`."""
    # Draw options
    kwargs, drawn = data.draw(option_contents_kwargs(), label='opts')

    allow_indexed_option = kwargs.get(
        'allow_indexed_option', DEFAULTS['allow_indexed_option']
    )
    allow_byte_masked = kwargs.get('allow_byte_masked', DEFAULTS['allow_byte_masked'])
    allow_bit_masked = kwargs.get('allow_bit_masked', DEFAULTS['allow_bit_masked'])
    allow_unmasked = kwargs.get('allow_unmasked', DEFAULTS['allow_unmasked'])

    # Call the test subject
    result = data.draw(st_ak.content.option_contents(**kwargs), label='result')

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
    if 'content' in kwargs:
        assert result.content is drawn['content']

    # Assert length is within bounds
    min_size = kwargs.get('min_size', DEFAULTS['min_size'])
    max_size = kwargs.get('max_size', DEFAULTS['max_size'])
    assert min_size <= len(result) <= sc(max_size)


@pytest.mark.parametrize('min_size', [1, 2, 10])
def test_draw_min_size(min_size: int) -> None:
    """Assert the result length can reach `min_size`."""
    find(
        st_ak.content.option_contents(min_size=min_size),
        lambda c: len(c) == min_size,
        settings=FIND,
    )


@pytest.mark.parametrize('cls', OPTION_TYPES)
def test_draw_option_type(cls: type[Content]) -> None:
    """Assert the given option type can be drawn."""
    find(st_ak.content.option_contents(), lambda c: isinstance(c, cls), settings=FIND)


def test_raise_all_disallowed() -> None:
    """Assert `ValueError` when every option type is disallowed."""
    with pytest.raises(
        ValueError, match='at least one option content type must be allowed'
    ):
        st_ak.content.option_contents(**{flag: False for flag in ALLOW_FLAGS})
