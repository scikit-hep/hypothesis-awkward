from typing import TypedDict

import numpy as np
import pytest
from hypothesis import find, given
from hypothesis import strategies as st

from awkward.contents import ByteMaskedArray, Content
from hypothesis_awkward import strategies as st_ak
from tests.find_settings import FIND
from tests.funcs import KwargsDrawn, assert_kwargs_match_signature


class ByteMaskedArrayContentsKwargs(TypedDict, total=False):
    """Options for `byte_masked_array_contents()` strategy."""

    content: st.SearchStrategy[Content] | Content | None


DEFAULTS = ByteMaskedArrayContentsKwargs(content=None)


class ByteMaskedArrayContentsDrawn(TypedDict):
    """Values drawn in advance for the options of `byte_masked_array_contents()`."""

    content: Content | None


def test_kwargs_match_signature() -> None:
    """Assert the option declarations agree with `byte_masked_array_contents()`."""
    assert_kwargs_match_signature(
        func=st_ak.constructors.byte_masked_array_contents,
        kwargs_cls=ByteMaskedArrayContentsKwargs,
        defaults=DEFAULTS,
    )


@st.composite
def byte_masked_array_contents_kwargs(
    draw: st.DrawFn,
) -> KwargsDrawn[ByteMaskedArrayContentsKwargs, ByteMaskedArrayContentsDrawn]:
    """Strategy for options for `byte_masked_array_contents()` strategy."""
    kwargs = ByteMaskedArrayContentsKwargs()
    drawn = ByteMaskedArrayContentsDrawn(content=None)
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
    """Assert the results of `byte_masked_array_contents()`."""
    # Draw options
    kwargs, drawn = data.draw(byte_masked_array_contents_kwargs(), label='opts')

    # Call the test subject
    result = data.draw(
        st_ak.constructors.byte_masked_array_contents(**kwargs), label='result'
    )

    assert isinstance(result, ByteMaskedArray)

    # Assert mask dtype is int8
    assert result.mask.data.dtype == np.int8

    # Assert mask length equals content length
    assert len(result.mask) == len(result.content)

    # Assert mask values are 0 or 1
    assert set(result.mask.data).issubset({0, 1})

    # Assert valid_when is bool
    assert isinstance(result.valid_when, bool)

    # Assert the options were effective
    if 'content' in kwargs:
        assert result.content is drawn['content']


@pytest.mark.parametrize('valid_when', [True, False])
def test_draw_valid_when(valid_when: bool) -> None:
    """Assert the given `valid_when` can be drawn."""
    find(
        st_ak.constructors.byte_masked_array_contents(),
        lambda c: c.valid_when is valid_when,
        settings=FIND,
    )


def test_draw_with_none_values() -> None:
    """Assert that masked (None) entries can appear."""
    find(
        st_ak.constructors.byte_masked_array_contents(),
        lambda c: (
            len(c) > 0 and any(c.mask.data[i] != c.valid_when for i in range(len(c)))
        ),
        settings=FIND,
    )


def test_draw_all_valid() -> None:
    """Assert that all-valid arrays can be drawn."""
    find(
        st_ak.constructors.byte_masked_array_contents(),
        lambda c: (
            len(c) > 0 and all(c.mask.data[i] == c.valid_when for i in range(len(c)))
        ),
        settings=FIND,
    )


def test_draw_from_contents() -> None:
    """Assert `contents()` can generate a `ByteMaskedArray` as outermost."""
    find(
        st_ak.constructors.contents(),
        lambda c: isinstance(c, ByteMaskedArray),
        settings=FIND,
    )
