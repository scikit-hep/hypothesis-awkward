import numpy as np
from hypothesis import assume
from hypothesis import strategies as st

import awkward as ak
from awkward.contents import BitMaskedArray, Content
from hypothesis_awkward._util import content_size

from ._types import StContent, StOption


@st.composite
def bit_masked_array_contents(
    draw: st.DrawFn,
    content: st.SearchStrategy[Content] | Content | None = None,
) -> BitMaskedArray:
    """Strategy for [`ak.contents.BitMaskedArray`][] instances.

    The logical length always equals `len(content)`. The mask is
    bit-packed into uint8 bytes.

    Parameters
    ----------
    content
        Child content. Can be a strategy for [`Content`][ak.contents.Content], a concrete
        [`Content`][ak.contents.Content] instance, or `None` to draw from
        `contents()`.

    Returns
    -------
    BitMaskedArray

    Examples
    --------
    >>> import awkward as ak
    >>> import numpy as np
    >>> from hypothesis import given
    >>> from hypothesis import strategies as st
    >>> import hypothesis_awkward.strategies as st_ak

    Give the content as a strategy, here for `NumpyArray`s of `int64`. The type is the
    option type of the content's type:

    >>> ints = st_ak.constructors.numpy_array_contents(
    ...     dtypes=st.just(np.dtype('int64'))
    ... )
    >>> @given(c=st_ak.constructors.bit_masked_array_contents(ints))
    ... def test_type(c):
    ...     assert str(ak.Array(c).type.content) == '?int64'
    >>> test_type()
    """
    # Imported here because content.py imports this module.
    from .content import contents

    match content:
        case None:
            content = draw(
                contents(
                    allow_union_root=False,
                    allow_option_root=False,
                    allow_indexed_root=False,
                )
            )
        case st.SearchStrategy():
            content = draw(content)
        case Content():
            pass
    assert isinstance(content, Content)
    length = len(content)
    n_bytes = (length + 7) // 8
    mask = draw(st.lists(st.integers(0, 255), min_size=n_bytes, max_size=n_bytes))
    mask_array = np.array(mask, dtype=np.uint8)
    valid_when = draw(st.booleans())
    lsb_order = draw(st.booleans())
    return BitMaskedArray(
        ak.index.IndexU8(mask_array), content, valid_when, length, lsb_order
    )


@st.composite
def bit_masked_array_from_contents(
    draw: st.DrawFn,
    content: StContent,
    *,
    max_size: int,
    max_leaf_size: int | None = None,
    min_length: int = 0,
    max_length: int | None = None,
    st_option: StOption | None = None,
) -> BitMaskedArray:
    """Strategy for [`ak.contents.BitMaskedArray`][] instances within a size budget.

    Called by `contents()` during recursive tree generation.

    Parameters
    ----------
    content
        A callable that accepts `max_size` and `max_leaf_size` and returns a strategy
        for a single content.
    max_size
        Upper bound on the number of scalars in the result, counted as for `max_size` in
        `contents()`.
    max_leaf_size
        Upper bound on the number of leaf elements in the result, counted as for
        `max_leaf_size` in `contents()`. Unbounded if `None`.
    min_length
        Lower bound on `len(result)`. Forwarded to the inner `content(...)` call
        so the inner content meets the floor.
    max_length
        Upper bound on `len(result)`. Unbounded if `None`.
    st_option
        Accepted for `_StFromContents` compatibility; unused in this variant.

    Returns
    -------
    BitMaskedArray

    Examples
    --------
    >>> from hypothesis_awkward.strategies.constructors import contents
    >>> c = bit_masked_array_from_contents(
    ...     contents, max_size=20, max_leaf_size=10, max_length=5
    ... ).example()
    >>> isinstance(c, BitMaskedArray)
    True

    >>> len(c) <= 5
    True
    """
    max_content_size = max(max_size - 2, 0)
    if max_length is not None:
        max_content_size = min(max_content_size, max_length)
    st_content = content(
        max_size=max_content_size,
        max_leaf_size=max_leaf_size,
        min_length=min_length,
        max_length=max_length,
        allow_option_root=False,
        allow_union_root=False,
        allow_indexed_root=False,
    )
    result = draw(bit_masked_array_contents(st_content))
    assume(content_size(result) <= max_size)
    return result
