from typing import TypedDict

from hypothesis import find, given
from hypothesis import strategies as st

from awkward.contents import Content, UnmaskedArray
from hypothesis_awkward import strategies as st_ak
from tests.find_settings import FIND
from tests.funcs import KwargsDrawn, assert_kwargs_match_signature


class UnmaskedArrayContentsKwargs(TypedDict, total=False):
    """Options for `unmasked_array_contents()` strategy."""

    content: st.SearchStrategy[Content] | Content | None


DEFAULTS = UnmaskedArrayContentsKwargs(content=None)


class UnmaskedArrayContentsDrawn(TypedDict):
    """Values drawn in advance for the options of `unmasked_array_contents()`."""

    content: Content | None


def test_kwargs_match_signature() -> None:
    """Assert the option declarations agree with `unmasked_array_contents()`."""
    assert_kwargs_match_signature(
        func=st_ak.constructors.unmasked_array_contents,
        kwargs_cls=UnmaskedArrayContentsKwargs,
        defaults=DEFAULTS,
    )


@st.composite
def unmasked_array_contents_kwargs(
    draw: st.DrawFn,
) -> KwargsDrawn[UnmaskedArrayContentsKwargs, UnmaskedArrayContentsDrawn]:
    """Strategy for options for `unmasked_array_contents()` strategy."""
    kwargs = UnmaskedArrayContentsKwargs()
    drawn = UnmaskedArrayContentsDrawn(content=None)
    if draw(st.booleans()):
        content = draw(
            st_ak.constructors.contents(
                allow_union_root=False,
                allow_option_root=False,
                allow_indexed_root=False,
            )
        )
        kwargs['content'] = draw(st.sampled_from([content, st.just(content)]))
        drawn['content'] = content
    return KwargsDrawn(kwargs, drawn)


@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """Assert the results of `unmasked_array_contents()`."""
    # Draw options
    kwargs, drawn = data.draw(unmasked_array_contents_kwargs(), label='opts')

    # Call the test subject
    result = data.draw(
        st_ak.constructors.unmasked_array_contents(**kwargs), label='result'
    )

    assert isinstance(result, UnmaskedArray)

    # Assert length equals content length
    assert len(result) == len(result.content)

    # Assert the options were effective
    if 'content' in kwargs:
        assert result.content is drawn['content']


def test_draw_nonempty() -> None:
    """Assert the length can be positive."""
    find(
        st_ak.constructors.unmasked_array_contents(),
        lambda c: len(c) > 0,
        settings=FIND,
    )


def test_draw_empty() -> None:
    """Assert the length can be zero."""
    find(
        st_ak.constructors.unmasked_array_contents(),
        lambda c: len(c) == 0,
        settings=FIND,
    )


def test_draw_from_contents() -> None:
    """Assert `contents()` can generate an `UnmaskedArray` as outermost."""
    find(
        st_ak.constructors.contents(),
        lambda c: isinstance(c, UnmaskedArray),
        settings=FIND,
    )
