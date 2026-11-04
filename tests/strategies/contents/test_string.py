from typing import TypedDict, cast

import numpy as np
from hypothesis import find, given
from hypothesis import strategies as st

import awkward as ak
from awkward.contents import ListOffsetArray, NumpyArray
from hypothesis_awkward import strategies as st_ak
from hypothesis_awkward.util import safe_compare as sc
from tests.find_settings import FIND
from tests.funcs import assert_kwargs_match_signature


class StringContentsKwargs(TypedDict, total=False):
    """Options for `string_contents()` strategy."""

    alphabet: st.SearchStrategy[str] | None
    min_size: int
    max_size: int


DEFAULTS = StringContentsKwargs(alphabet=None, min_size=0, max_size=10)


class RelatedKwargs(TypedDict, total=False):
    """Options drawn together as an ordered pair (`min_size <= max_size`)."""

    min_size: int
    max_size: int


def test_kwargs_match_signature() -> None:
    """Assert the option declarations agree with `string_contents()`."""
    assert_kwargs_match_signature(
        func=st_ak.contents.string_contents,
        kwargs_cls=StringContentsKwargs,
        defaults=DEFAULTS,
        related_cls=RelatedKwargs,
    )


@st.composite
def string_contents_kwargs(draw: st.DrawFn) -> StringContentsKwargs:
    """Strategy for options for `string_contents()` strategy."""

    @st.composite
    def _st_related_kwargs(draw: st.DrawFn) -> RelatedKwargs:
        """Strategy for the options that are drawn together."""
        min_size, max_size = draw(
            st_ak.ranges(min_start=0, max_start=DEFAULTS['max_size'], max_end=100)
        )
        ret = RelatedKwargs()
        if min_size is not None:
            ret['min_size'] = min_size
        if max_size is not None:
            ret['max_size'] = max_size
        return ret

    related = draw(_st_related_kwargs())

    optional_independent = draw(
        st.fixed_dictionaries(
            {},
            optional={
                'alphabet': st.one_of(
                    st.none(),
                    st.just(st_ak.RecordDraws(st.characters(codec='utf-8'))),
                ),
            },
        )
    )

    return StringContentsKwargs(
        **related, **cast(StringContentsKwargs, optional_independent)
    )


@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """Assert the results of `string_contents()`."""
    # Draw options
    kwargs = data.draw(string_contents_kwargs(), label='kwargs')

    # Call the test subject
    result = data.draw(st_ak.contents.string_contents(**kwargs), label='result')

    # Assert the result is always a ListOffsetArray
    assert isinstance(result, ListOffsetArray)

    # Assert outer __array__ parameter
    assert result.parameter('__array__') == 'string'

    # Assert inner content is NumpyArray
    assert isinstance(result.content, NumpyArray)

    # Assert inner __array__ parameter
    assert result.content.parameter('__array__') == 'char'

    # Assert inner dtype is uint8
    assert result.content.dtype == np.dtype(np.uint8)

    # Assert size bounds
    alphabet = kwargs.get('alphabet', DEFAULTS['alphabet'])
    min_size = kwargs.get('min_size', DEFAULTS['min_size'])
    max_size = kwargs.get('max_size', DEFAULTS['max_size'])

    assert sc(min_size) <= len(result) <= sc(max_size)

    # Assert all strings are valid UTF-8
    arr = ak.Array(result)
    for s in arr.to_list():
        assert isinstance(s, str)
        s.encode('utf-8')

    # Assert alphabet constraint
    match alphabet:
        case st_ak.RecordDraws():
            drawn_chars = set(alphabet.drawn)
            for s in arr.to_list():
                for ch in s:
                    assert ch in drawn_chars


def test_draw_empty() -> None:
    """Assert that empty arrays (zero strings) can be drawn."""
    find(st_ak.contents.string_contents(), lambda c: len(c) == 0, settings=FIND)


def test_draw_max_size() -> None:
    """Assert that arrays with exactly max_size strings can be drawn."""
    find(
        st_ak.contents.string_contents(),
        lambda c: len(c) == DEFAULTS['max_size'],
        settings=FIND,
    )


def test_draw_empty_string() -> None:
    """Assert that arrays containing an empty string can be drawn."""
    find(
        st_ak.contents.string_contents(min_size=1),
        lambda c: '' in ak.Array(c).to_list(),
        settings=FIND,
    )


def test_draw_non_ascii() -> None:
    """Assert that arrays containing non-ASCII characters can be drawn."""
    find(
        st_ak.contents.string_contents(min_size=1),
        lambda c: any(not s.isascii() for s in ak.Array(c).to_list()),
        settings=FIND,
    )
