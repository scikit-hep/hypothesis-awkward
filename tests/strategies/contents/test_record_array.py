from typing import TypedDict, cast

import pytest
from hypothesis import find, given
from hypothesis import strategies as st

from awkward.contents import Content, RecordArray
from hypothesis_awkward import strategies as st_ak
from hypothesis_awkward._util import safe_compare as sc
from tests.find_settings import FIND
from tests.funcs import KwargsDrawn, assert_kwargs_match_signature


class RecordArrayContentsKwargs(TypedDict, total=False):
    """Options for `record_array_contents()` strategy."""

    contents: list[Content] | st.SearchStrategy[list[Content]] | None
    max_fields: int
    allow_tuple: bool
    min_length: int
    max_length: int | None


DEFAULTS = RecordArrayContentsKwargs(
    contents=None,
    max_fields=5,
    allow_tuple=True,
    min_length=0,
    max_length=None,
)


class RecordArrayContentsDrawn(TypedDict, total=False):
    """Values drawn in advance for the options of `record_array_contents()`."""

    contents: list[Content]


class RelatedKwargs(TypedDict, total=False):
    """Options drawn together as an ordered pair (`min_length <= max_length`)."""

    min_length: int
    max_length: int | None


def test_kwargs_match_signature() -> None:
    """Assert the option declarations agree with `record_array_contents()`."""
    assert_kwargs_match_signature(
        func=st_ak.contents.record_array_contents,
        kwargs_cls=RecordArrayContentsKwargs,
        defaults=DEFAULTS,
        related_cls=RelatedKwargs,
    )


@st.composite
def _contents_list(
    draw: st.DrawFn,
) -> list[Content]:
    """Draw a list of 1..5 Content objects for testing."""
    n = draw(st.integers(min_value=1, max_value=5))
    return [
        draw(st_ak.contents.contents(max_leaf_size=5, max_depth=2)) for _ in range(n)
    ]


@st.composite
def record_array_contents_kwargs(
    draw: st.DrawFn,
) -> KwargsDrawn[RecordArrayContentsKwargs, RecordArrayContentsDrawn]:
    """Strategy for options for `record_array_contents()` strategy."""

    @st.composite
    def _st_related_kwargs(draw: st.DrawFn) -> RelatedKwargs:
        """Strategy for the options that are drawn together."""
        min_length, max_length = draw(st_ak.ranges(min_start=0, max_end=5))
        ret = RelatedKwargs()
        if min_length is not None:
            ret['min_length'] = min_length
        if max_length is not None:
            ret['max_length'] = max_length
        return ret

    @st.composite
    def _st_drawn(draw: st.DrawFn) -> RecordArrayContentsDrawn:
        """Strategy for the values drawn in advance for the options."""
        ret = RecordArrayContentsDrawn()
        if draw(st.booleans()):
            ret['contents'] = draw(_contents_list())
        return ret

    @st.composite
    def _st_optional_independent(
        draw: st.DrawFn, drawn: RecordArrayContentsDrawn
    ) -> RecordArrayContentsKwargs:
        """Strategy for the options that are drawn independently, if at all."""
        ret = cast(
            RecordArrayContentsKwargs,
            draw(
                st.fixed_dictionaries(
                    {},
                    optional={
                        'max_fields': st.integers(min_value=0, max_value=10),
                        'allow_tuple': st.booleans(),
                    },
                )
            ),
        )
        if 'contents' in drawn:
            contents = drawn['contents']
            choices = list[list[Content] | st.SearchStrategy[list[Content]]](
                [contents, st.just(contents)]
            )
            ret['contents'] = draw(st.sampled_from(choices))
        return ret

    related = draw(_st_related_kwargs())
    drawn = draw(_st_drawn())
    optional_independent = draw(_st_optional_independent(drawn))

    kwargs = RecordArrayContentsKwargs(**related, **optional_independent)
    return KwargsDrawn(kwargs, drawn)


@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """Assert the results of `record_array_contents()`."""
    # Draw options
    kwargs, drawn = data.draw(record_array_contents_kwargs(), label='opts')

    # Call the test subject
    result = data.draw(st_ak.contents.record_array_contents(**kwargs), label='result')

    assert isinstance(result, RecordArray)

    # Assert the options were effective
    max_fields = kwargs.get('max_fields', DEFAULTS['max_fields'])
    allow_tuple = kwargs.get('allow_tuple', DEFAULTS['allow_tuple'])

    if 'contents' in kwargs:
        # Given contents: the fields are the same objects
        assert len(result.contents) == len(drawn['contents'])
        for r_content, drawn_content in zip(result.contents, drawn['contents']):
            assert r_content is drawn_content
    else:
        # When auto-generating contents, field count is bounded by max_fields
        assert len(result.contents) <= max_fields

    # allow_tuple=False means the result must be a named record
    if not allow_tuple:
        assert not result.is_tuple

    # Named records have unique field names matching the number of contents
    if not result.is_tuple:
        assert len(result.fields) == len(result.contents)
        assert len(set(result.fields)) == len(result.fields)

    # Assert length is within bounds
    min_length = kwargs.get('min_length', DEFAULTS['min_length'])
    max_length = kwargs.get('max_length', DEFAULTS['max_length'])
    assert min_length <= len(result) <= sc(max_length)

    # Non-empty records: length equals the minimum content length, capped by max_length
    if result.contents:
        expected = min(len(c) for c in result.contents)
        if max_length is not None:
            expected = min(expected, max_length)
        assert result.length == expected


def test_draw_tuple() -> None:
    """Assert the record can be a tuple."""
    find(st_ak.contents.record_array_contents(), lambda r: r.is_tuple, settings=FIND)


def test_draw_named() -> None:
    """Assert the record can be named."""
    find(
        st_ak.contents.record_array_contents(), lambda r: not r.is_tuple, settings=FIND
    )


@pytest.mark.parametrize('max_fields', [0, 1, 2, 5])
def test_draw_max_fields(max_fields: int) -> None:
    """Assert the field count can reach `max_fields`."""
    find(
        st_ak.contents.record_array_contents(max_fields=max_fields),
        lambda r: len(r.contents) == max_fields,
        settings=FIND,
    )


@pytest.mark.parametrize('min_length', [1, 2, 10])
def test_draw_min_length(min_length: int) -> None:
    """Assert the length can reach `min_length`."""
    find(
        st_ak.contents.record_array_contents(min_length=min_length),
        lambda r: len(r) == min_length,
        settings=FIND,
    )


@pytest.mark.parametrize('max_length', [0, 1, 2, 4])
@pytest.mark.parametrize('is_tuple', [True, False])
@pytest.mark.parametrize('with_contents', [True, False])
def test_draw_max_length(max_length: int, is_tuple: bool, with_contents: bool) -> None:
    """Assert the length can reach `max_length`."""
    find(
        st_ak.contents.record_array_contents(max_length=max_length),
        lambda r: (
            bool(r.contents) == with_contents
            and r.is_tuple == is_tuple
            and len(r) == max_length
        ),
        settings=FIND,
    )


@pytest.mark.parametrize('is_tuple', [True, False])
def test_draw_zero_fields(is_tuple: bool) -> None:
    """Assert a zero-field record can have a non-zero length."""
    find(
        st_ak.contents.record_array_contents(),
        lambda r: not r.contents and r.is_tuple == is_tuple and len(r) != 0,
        settings=FIND,
    )


def test_draw_from_contents() -> None:
    """Assert `contents()` can generate a `RecordArray` as outermost."""
    find(st_ak.contents.contents(), lambda c: isinstance(c, RecordArray), settings=FIND)
