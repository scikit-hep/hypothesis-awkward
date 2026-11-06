import functools
import itertools
from typing import Any, TypedDict, cast

import pytest
from hypothesis import find, given
from hypothesis import strategies as st

from awkward._do import mergeable
from awkward.contents import Content, IndexedArray, UnionArray
from hypothesis_awkward import strategies as st_ak
from hypothesis_awkward.strategies import RecordCallDraws
from hypothesis_awkward.strategies.contents.option import StOption, option_from_contents
from hypothesis_awkward.strategies.contents.union_array import (
    union_array_from_contents,
)
from hypothesis_awkward.util import content_size, leaf_size
from hypothesis_awkward.util import safe_compare as sc
from tests.find_settings import FIND, FIND_RARE_NO_SHRINK
from tests.funcs import assert_kwargs_match_signature


class UnionArrayFromContentsKwargs(TypedDict, total=False):
    """Options for `union_array_from_contents()` strategy."""

    max_leaf_size: int | None
    min_length: int
    max_length: int | None
    allow_invalid: bool
    st_option: StOption | None


DEFAULTS = UnionArrayFromContentsKwargs(
    max_leaf_size=None,
    min_length=0,
    max_length=None,
    allow_invalid=True,
    st_option=None,
)


class RelatedKwargs(TypedDict, total=False):
    """Options drawn together as an ordered pair (`min_length <= max_length`)."""

    min_length: int
    max_length: int | None


def test_kwargs_match_signature() -> None:
    """Assert the option declarations agree with `union_array_from_contents()`."""
    assert_kwargs_match_signature(
        func=union_array_from_contents,
        exclude={'content', 'max_size'},
        kwargs_cls=UnionArrayFromContentsKwargs,
        defaults=DEFAULTS,
        related_cls=RelatedKwargs,
    )


@st.composite
def union_array_from_contents_kwargs(
    draw: st.DrawFn, *, allow_option: bool
) -> UnionArrayFromContentsKwargs:
    """Strategy for options for `union_array_from_contents()` strategy.

    Children that can be option types need `st_option`, without which they could
    mix option and non-option types, which `UnionArray` rejects.
    """

    @st.composite
    def _st_related_kwargs(draw: st.DrawFn) -> RelatedKwargs:
        """Strategy for the options that are drawn together."""
        min_length, max_length = draw(st_ak.ranges(min_start=0, max_end=10))
        ret = RelatedKwargs()
        if min_length is not None:
            ret['min_length'] = min_length
        if max_length is not None:
            ret['max_length'] = max_length
        return ret

    related = draw(_st_related_kwargs())

    optional_independent = cast(
        UnionArrayFromContentsKwargs,
        draw(
            st.fixed_dictionaries(
                {},
                optional={
                    'max_leaf_size': st.one_of(
                        st.none(), st.integers(min_value=0, max_value=20)
                    ),
                    'allow_invalid': st.booleans(),
                },
            )
        ),
    )
    if allow_option:
        optional_independent['st_option'] = option_from_contents
    elif draw(st.booleans()):
        optional_independent['st_option'] = None

    return UnionArrayFromContentsKwargs(**related, **optional_independent)


@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """Assert the results of `union_array_from_contents()`."""
    # Draw options
    allow_option = data.draw(st.booleans(), label='allow_option')
    content = RecordCallDraws(
        st_ak.contents.contents if allow_option else _contents_with_no_option
    )
    max_size = data.draw(st.integers(min_value=0, max_value=50), label='max_size')
    kwargs = data.draw(
        union_array_from_contents_kwargs(allow_option=allow_option), label='kwargs'
    )

    # Call the test subject
    result = data.draw(
        union_array_from_contents(content, max_size=max_size, **kwargs), label='result'
    )

    assert isinstance(result, UnionArray)
    children = result.contents
    assert len(children) >= 2

    # Assert no child is of a type that `UnionArray` does not allow as a child
    assert not any(isinstance(c, (UnionArray, IndexedArray)) for c in children)

    # Assert the children are all option types or none is
    assert len({c.is_option for c in children}) == 1

    # Assert the options were effective
    assert content_size(result) <= max_size

    max_leaf_size = kwargs.get('max_leaf_size', DEFAULTS['max_leaf_size'])
    assert leaf_size(result) <= sc(max_leaf_size)

    min_length = kwargs.get('min_length', DEFAULTS['min_length'])
    max_length = kwargs.get('max_length', DEFAULTS['max_length'])
    assert min_length <= len(result) <= sc(max_length)

    allow_invalid = kwargs.get('allow_invalid', DEFAULTS['allow_invalid'])
    if not allow_invalid:
        # TODO: Assert `ak.validity_error(result) == ''` when `contents()` has the
        # option. Until then, the children can contain invalid nested unions.
        assert not _has_mergeable_contents(result)

    # Assert the children are what `content` drew. Those wrapped in an option type
    # by `st_option` are not the same objects as the draws. Mergeable draws can be
    # dropped if not `allow_invalid`.
    if allow_invalid:
        assert len(content.drawn) == len(children)
    if not allow_option:
        drawn_ids = [id(d) for d in content.drawn]
        assert all(id(c) in drawn_ids for c in children)


@pytest.mark.parametrize('n_contents', [2, 3])
def test_draw_n_contents(n_contents: int) -> None:
    """Assert `n_contents` child contents can be drawn."""
    find(_st_union(), lambda u: len(u.contents) == n_contents, settings=FIND)


def test_draw_all_option() -> None:
    """Assert all-option children can be drawn."""
    find(
        _st_union(),
        lambda u: all(c.is_option for c in u.contents),
        settings=FIND_RARE_NO_SHRINK,
    )


def test_draw_mergeable_contents() -> None:
    """Assert mergeable child contents can be drawn by default."""
    find(_st_union(), _has_mergeable_contents, settings=FIND)


@pytest.mark.parametrize('n_contents', [2, 3])
def test_draw_n_contents_not_allow_invalid(n_contents: int) -> None:
    """Assert `n_contents` child contents can be drawn if not `allow_invalid`."""
    find(
        _st_union(allow_invalid=False),
        lambda u: len(u.contents) == n_contents,
        settings=FIND,
    )


def test_draw_all_option_not_allow_invalid() -> None:
    """Assert all-option children can be drawn if not `allow_invalid`."""
    find(
        _st_union(allow_invalid=False),
        lambda u: all(c.is_option for c in u.contents),
        settings=FIND_RARE_NO_SHRINK,
    )


def _contents_with_no_option(**kwargs: Any) -> st.SearchStrategy[Content]:
    return st_ak.contents.contents(
        allow_indexed_option=False,
        allow_byte_masked=False,
        allow_bit_masked=False,
        allow_unmasked=False,
        **kwargs,
    )


_st_union = functools.partial(
    union_array_from_contents,
    st_ak.contents.contents,
    max_size=20,
    max_leaf_size=10,
    st_option=option_from_contents,
)


def _has_mergeable_contents(u: UnionArray) -> bool:
    pairs = itertools.combinations(u.contents, 2)
    return any(mergeable(one, two, mergebool=False) for one, two in pairs)
