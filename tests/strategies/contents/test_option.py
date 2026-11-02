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
from hypothesis_awkward.util import safe_compare as sc
from tests.find_settings import FIND
from tests.funcs import assert_kwargs_match_signature

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
        func=st_ak.contents.option_contents,
        kwargs_cls=OptionContentsKwargs,
        defaults=DEFAULTS,
        related_cls=RelatedKwargs,
    )


@st.composite
def option_contents_kwargs(
    draw: st.DrawFn,
    chain: st_ak.OptsChain[Any] | None = None,
) -> st_ak.OptsChain[OptionContentsKwargs]:
    """Strategy for options for `option_contents()` strategy."""
    if chain is None:
        chain = st_ak.OptsChain({})
    st_content = chain.register(
        st_ak.contents.contents(
            allow_union_root=False, allow_option_root=False, allow_indexed_root=False
        )
    )

    @st.composite
    def _st_related_kwargs(draw: st.DrawFn) -> RelatedKwargs:
        """Strategy for the options that are drawn together."""
        min_size, max_size = draw(st_ak.ranges(min_start=0, max_end=50))
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

    optional_independent = draw(
        st.fixed_dictionaries(
            {},
            optional={
                'content': st.one_of(
                    st_ak.contents.contents(
                        allow_union_root=False,
                        allow_option_root=False,
                        allow_indexed_root=False,
                    ),
                    st.just(st_content),
                ),
            },
        )
    )

    return chain.extend(cast(OptionContentsKwargs, {**related, **optional_independent}))


@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """Assert the results of `option_contents()`."""
    # Draw options
    opts = data.draw(option_contents_kwargs(), label='opts')
    opts.reset()

    allow_indexed_option = opts.kwargs.get(
        'allow_indexed_option', DEFAULTS['allow_indexed_option']
    )
    allow_byte_masked = opts.kwargs.get(
        'allow_byte_masked', DEFAULTS['allow_byte_masked']
    )
    allow_bit_masked = opts.kwargs.get('allow_bit_masked', DEFAULTS['allow_bit_masked'])
    allow_unmasked = opts.kwargs.get('allow_unmasked', DEFAULTS['allow_unmasked'])

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
    content = opts.kwargs.get('content', DEFAULTS['content'])
    match content:
        case Content():
            assert result.content is content
        case st_ak.RecordDraws():
            assert len(content.drawn) == 1
            assert result.content is content.drawn[0]

    # Assert length is within bounds
    min_size = opts.kwargs.get('min_size', DEFAULTS['min_size'])
    max_size = opts.kwargs.get('max_size', DEFAULTS['max_size'])
    assert min_size <= len(result) <= sc(max_size)


@pytest.mark.parametrize('min_size', [1, 2, 10])
def test_draw_min_size(min_size: int) -> None:
    """Assert the result length can reach `min_size`."""
    find(
        st_ak.contents.option_contents(min_size=min_size),
        lambda c: len(c) == min_size,
        settings=FIND,
    )


@pytest.mark.parametrize('cls', OPTION_TYPES)
def test_draw_option_type(cls: type[Content]) -> None:
    """Assert the given option type can be drawn."""
    find(st_ak.contents.option_contents(), lambda c: isinstance(c, cls), settings=FIND)


def test_raise_all_disallowed() -> None:
    """Assert `ValueError` when every option type is disallowed."""
    with pytest.raises(
        ValueError, match='at least one option content type must be allowed'
    ):
        st_ak.contents.option_contents(**{flag: False for flag in ALLOW_FLAGS})
