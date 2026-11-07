import numpy as np
from hypothesis import assume
from hypothesis import strategies as st

import awkward as ak
from awkward.contents import (
    Content,
    EmptyArray,
    IndexedArray,
    IndexedOptionArray,
    NumpyArray,
)
from hypothesis_awkward import strategies as st_ak
from hypothesis_awkward._util import content_size, is_bytestring_leaf, is_string_leaf

from ._types import StContent, StOption

_CATEGORICAL = {'__array__': 'categorical'}


@st.composite
def categorical_contents(
    draw: st.DrawFn,
    content: st.SearchStrategy[Content] | Content | None = None,
    *,
    min_size: int = 0,
    max_size: int | None = None,
    allow_invalid: bool = True,
    allow_indexed: bool = True,
    allow_indexed_option: bool = True,
) -> IndexedArray | IndexedOptionArray:
    """Strategy for categorical data.

    Categorical data is an [`IndexedArray`][ak.contents.IndexedArray] or an
    [`IndexedOptionArray`][ak.contents.IndexedOptionArray] with the parameter
    `__array__` set to `'categorical'`. Its content holds the categories, and the
    index refers to them. The categories are a 1-D
    [`NumpyArray`][ak.contents.NumpyArray], an
    [`EmptyArray`][ak.contents.EmptyArray], string content, or bytestring content.

    Parameters
    ----------
    content
        The categories. Can be a strategy for [`Content`][ak.contents.Content], a
        concrete [`Content`][ak.contents.Content] instance, or `None` to draw from
        `leaf_contents()`. They must be of a kind listed above and, if `allow_invalid`
        is `False`, unique. A concrete `content` that is not raises `ValueError`; draws
        from a strategy that are not are rejected. Empty categories can be referred
        to only by missing entries, so they are rejected if `min_size > 0` and
        `allow_indexed_option` is `False`.
    min_size
        Lower bound on the index length, i.e., `len(result)`.
    max_size
        Upper bound on the index length, i.e., `len(result)`. If `None`, twice the
        content length is used.
    allow_invalid
        Duplicate categories can occur if `True`, which [`ak.validity_error`][] and
        [`ak.is_valid`][] consider invalid. No two categories are equal if `False`, as
        Awkward Array defines it: `0.0` and `-0.0` are equal, `NaN`s are distinct, and
        at most one `NaT` occurs. A `content` of a kind not listed above is rejected
        regardless of this option.
    allow_indexed
        No [`IndexedArray`][ak.contents.IndexedArray] is generated if `False`.
    allow_indexed_option
        No [`IndexedOptionArray`][ak.contents.IndexedOptionArray] is generated if
        `False`.

    Raises
    ------
    ValueError
        If both `allow_indexed` and `allow_indexed_option` are `False`, or if a
        concrete `content` is not valid as categories.

    Returns
    -------
    IndexedArray | IndexedOptionArray

    Examples
    --------
    >>> c = categorical_contents().example()
    >>> c.parameter('__array__')
    'categorical'
    """
    if not (allow_indexed or allow_indexed_option):
        raise ValueError(
            'at least one of allow_indexed and allow_indexed_option must be True'
        )

    unique = not allow_invalid

    match content:
        case None:
            # Without missing entries, a non-empty index needs a category to refer to.
            min_categories = 1 if min_size > 0 and not allow_indexed_option else 0
            st_categories = st_ak.contents.leaf_contents(
                min_size=min_categories, unique=unique, allow_zero_field_record=False
            )
            if unique:
                # Hypothesis counts `NaT`s as distinct; Awkward Array does not.
                st_categories = st_categories.filter(
                    lambda c: _is_valid_categories(c, unique=True)
                )
            categories = draw(st_categories)
        case st.SearchStrategy():
            categories = draw(content)
            assume(_is_valid_categories(categories, unique=unique))
        case Content():
            if not _is_valid_categories(content, unique=unique):
                raise ValueError(f'content is not valid as categories: {content!r}')
            categories = content

    # An empty content can be referred to only by missing entries.
    can_index = len(categories) > 0 or min_size == 0
    flavors = list[str]()
    if allow_indexed and can_index:
        flavors.append('indexed')
    if allow_indexed_option:
        flavors.append('indexed_option')
    assume(flavors)

    match draw(st.sampled_from(flavors)):
        case 'indexed':
            c = draw(
                st_ak.contents.indexed_array_contents(
                    categories, min_size=min_size, max_size=max_size
                )
            )
            return IndexedArray(c.index, c.content, parameters=_CATEGORICAL)
        case _:
            o = draw(
                st_ak.contents.indexed_option_array_contents(
                    categories, min_size=min_size, max_size=max_size
                )
            )
            return IndexedOptionArray(o.index, o.content, parameters=_CATEGORICAL)


@st.composite
def categorical_from_contents(
    draw: st.DrawFn,
    content: StContent,
    *,
    max_size: int,
    max_leaf_size: int | None = None,
    min_length: int = 0,
    max_length: int | None = None,
    allow_invalid: bool = True,
    allow_indexed: bool = True,
    allow_indexed_option: bool = True,
    st_option: StOption | None = None,
) -> IndexedArray | IndexedOptionArray:
    """Strategy for categorical data within a size budget.

    Draws the index length first, then gives the remainder of the budget to the
    categories.

    Called by `contents()` during recursive tree generation.

    Parameters
    ----------
    content
        A callable that accepts `max_size`, `max_leaf_size`, and `min_length` and
        returns a strategy for the categories. Draws that are not valid as categories
        are rejected, as in `categorical_contents()`.
    max_size
        Upper bound on the number of scalars in the result, counted as for `max_size` in
        `contents()`.
    max_leaf_size
        Upper bound on the number of leaf elements in the result, counted as for
        `max_leaf_size` in `contents()`. Unbounded if `None`.
    min_length
        Lower bound on `len(result)`.
    max_length
        Upper bound on `len(result)`. Unbounded if `None`.
    allow_invalid
        Duplicate categories can occur if `True`, which [`ak.validity_error`][] and
        [`ak.is_valid`][] consider invalid. Non-unique draws from `content` are
        rejected if `False`.
    allow_indexed
        No [`IndexedArray`][ak.contents.IndexedArray] is generated if `False`.
    allow_indexed_option
        No [`IndexedOptionArray`][ak.contents.IndexedOptionArray] is generated if
        `False`.
    st_option
        Accepted for `_StFromContents` compatibility; unused in this strategy.

    Raises
    ------
    ValueError
        If both `allow_indexed` and `allow_indexed_option` are `False`.

    Returns
    -------
    IndexedArray | IndexedOptionArray

    Examples
    --------
    >>> import functools
    >>> leaves = functools.partial(
    ...     st_ak.contents.contents, max_depth=0, allow_record=False
    ... )
    >>> c = categorical_from_contents(
    ...     leaves, max_size=20, min_length=2, max_length=5
    ... ).example()
    >>> c.parameter('__array__')
    'categorical'

    >>> 2 <= len(c) <= 5
    True
    """
    if not (allow_indexed or allow_indexed_option):
        raise ValueError(
            'at least one of allow_indexed and allow_indexed_option must be True'
        )

    ml = max_length if max_length is not None else max_size
    assume(min_length <= ml)
    n = draw(st.integers(min_value=min_length, max_value=ml))
    max_content_size = max(max_size - n, 0)
    # Without missing entries, a non-empty index needs a category to refer to.
    min_categories = 1 if n > 0 and not allow_indexed_option else 0
    assume(max_content_size >= min_categories)
    categories = draw(
        content(
            max_size=max_content_size,
            max_leaf_size=max_leaf_size,
            min_length=min_categories,
        )
    )
    result = draw(
        categorical_contents(
            st.just(categories),
            min_size=n,
            max_size=n,
            allow_invalid=allow_invalid,
            allow_indexed=allow_indexed,
            allow_indexed_option=allow_indexed_option,
        )
    )
    assume(content_size(result) <= max_size)
    return result


def _is_valid_categories(c: Content, *, unique: bool) -> bool:
    """Return True if `c` is of a kind of categories and, if `unique`, unique.

    Uniqueness follows Awkward Array: `0.0` and `-0.0` are equal, `NaN`s are distinct,
    and `NaT`s are equal. It is checked here rather than with `ak.is_valid()`, which
    fails on some dtypes.
    """
    match c:
        case EmptyArray():
            return True
        case NumpyArray():
            data = np.asarray(c.data)
            if data.ndim != 1:
                return False
            if not unique:
                return True
            if data.dtype.kind in 'fc':
                data = data[~np.isnan(data)]
            # `np.unique()` merges `0.0` and `-0.0`, and all `NaT`s.
            return len(np.unique(data)) == len(data)
        case _ if is_string_leaf(c) or is_bytestring_leaf(c):
            if not unique:
                return True
            values = ak.to_list(c)
            return len(set(values)) == len(values)
        case _:
            return False
