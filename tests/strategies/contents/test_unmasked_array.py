from typing import Any, TypedDict, cast

from hypothesis import Phase, find, given, settings
from hypothesis import strategies as st

from awkward.contents import Content, UnmaskedArray
from hypothesis_awkward import strategies as st_ak


class UnmaskedArrayContentsKwargs(TypedDict, total=False):
    """Options for `unmasked_array_contents()` strategy."""

    content: st.SearchStrategy[Content] | Content


@st.composite
def unmasked_array_contents_kwargs(
    draw: st.DrawFn,
    chain: st_ak.OptsChain[Any] | None = None,
) -> st_ak.OptsChain[UnmaskedArrayContentsKwargs]:
    """Strategy for options for `unmasked_array_contents()` strategy."""
    if chain is None:
        chain = st_ak.OptsChain({})
    st_content = chain.register(
        st_ak.contents.contents(allow_union_root=False, allow_option_root=False)
    )

    kwargs = draw(
        st.fixed_dictionaries(
            {},
            optional={
                'content': st.one_of(
                    st_ak.contents.contents(
                        allow_union_root=False, allow_option_root=False
                    ),
                    st.just(st_content),
                ),
            },
        )
    )

    return chain.extend(cast(UnmaskedArrayContentsKwargs, kwargs))


@settings(max_examples=200)
@given(data=st.data())
def test_unmasked_array_contents(data: st.DataObject) -> None:
    """Test that `unmasked_array_contents()` respects all its options."""
    # Draw options
    opts = data.draw(unmasked_array_contents_kwargs(), label='opts')
    opts.reset()

    # Call the test subject
    result = data.draw(
        st_ak.contents.unmasked_array_contents(**opts.kwargs), label='result'
    )

    assert isinstance(result, UnmaskedArray)

    # Assert length equals content length
    assert len(result) == len(result.content)

    # Assert the options were effective
    content = opts.kwargs.get('content', None)
    match content:
        case Content():
            assert result.content is content
        case st_ak.RecordDraws():
            assert len(content.drawn) == 1
            assert result.content is content.drawn[0]


def test_draw_from_contents() -> None:
    """Assert that UnmaskedArray can be the root node from `contents()`."""
    find(
        st_ak.contents.contents(),
        lambda c: isinstance(c, UnmaskedArray),
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_nonempty() -> None:
    """Assert that non-empty UnmaskedArray can be drawn."""
    find(
        st_ak.contents.unmasked_array_contents(),
        lambda c: len(c) > 0,
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_empty() -> None:
    """Assert that empty UnmaskedArray can be drawn."""
    find(
        st_ak.contents.unmasked_array_contents(),
        lambda c: len(c) == 0,
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )
