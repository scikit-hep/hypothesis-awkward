from hypothesis import strategies as st

from awkward.contents import Content, IndexedArray, IndexedOptionArray


@st.composite
def categorical_contents(
    draw: st.DrawFn,
    content: st.SearchStrategy[Content] | Content | None = None,
    *,
    min_size: int = 0,
    max_size: int | None = None,
    unique: bool = False,
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
        `leaf_contents()`. They must be of a kind listed above and, if `unique` is
        `True`, unique. A concrete `content` that is not raises `ValueError`; draws
        from a strategy that are not are rejected.
    min_size
        Lower bound on the index length, i.e., `len(result)`.
    max_size
        Upper bound on the index length, i.e., `len(result)`. If `None`, twice the
        content length is used.
    unique
        No two categories are equal if `True`, as Awkward Array defines it: `0.0` and
        `-0.0` are equal, `NaN`s are distinct, and at most one `NaT` occurs.
        [`ak.validity_error`][] and [`ak.is_valid`][] consider categorical data with
        duplicate categories invalid.
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
    raise NotImplementedError
