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
from hypothesis_awkward.util import is_bytestring_leaf, is_string_leaf

CATEGORICAL = {'__array__': 'categorical'}


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
                min_size=min_categories, unique=True, allow_zero_field_record=False
            )
            # Hypothesis counts `NaT`s as distinct; Awkward Array does not.
            categories = draw(
                st_categories.filter(lambda c: _is_valid_categories(c, unique=True))
            )
            if not unique:
                categories = draw(_with_duplicates(categories))
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
            return IndexedArray(c.index, c.content, parameters=CATEGORICAL)
        case _:
            o = draw(
                st_ak.contents.indexed_option_array_contents(
                    categories, min_size=min_size, max_size=max_size
                )
            )
            return IndexedOptionArray(o.index, o.content, parameters=CATEGORICAL)


@st.composite
def _with_duplicates(draw: st.DrawFn, categories: Content) -> Content:
    """Strategy for `categories`, distinct ones, with possible duplicates.

    Each duplicate overwrites a distinct category or, for `bool`, which has only two
    values, is appended. The distinct categories stay among the draws, so each
    duplicate shrinks away on its own and the result shrinks to valid categories
    unless the duplicates are needed.
    """
    n = len(categories)
    if n < 2 or not draw(st.booleans()):
        return categories
    index = list(range(n))
    for i in range(1, n):
        if draw(st.booleans()):
            index[i] = draw(st.integers(min_value=0, max_value=i - 1))
    if isinstance(categories, NumpyArray) and categories.data.dtype == np.bool_:
        index += draw(st.lists(st.integers(min_value=0, max_value=1), max_size=8))
    if index == list(range(n)):
        return categories
    duplicated = categories[np.array(index)]
    return duplicated.to_ListOffsetArray64(True) if duplicated.is_list else duplicated


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
