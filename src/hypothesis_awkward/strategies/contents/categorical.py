from hypothesis import strategies as st

from awkward.contents import Content, IndexedArray, IndexedOptionArray


@st.composite
def categorical_contents(
    draw: st.DrawFn,
    content: st.SearchStrategy[Content] | Content | None = None,
    *,
    allow_indexed: bool = True,
    allow_indexed_option: bool = True,
    min_size: int = 0,
    max_size: int | None = None,
) -> IndexedArray | IndexedOptionArray:
    """Strategy for categorical data.

    Categorical data is an [`IndexedArray`][ak.contents.IndexedArray] or an
    [`IndexedOptionArray`][ak.contents.IndexedOptionArray] with the parameter
    `__array__` set to `'categorical'`. Its content holds the categories, which are
    unique as Awkward Array defines it: `0.0` and `-0.0` are equal, `NaN`s are
    distinct, and at most one `NaT` occurs. The index refers to the categories.

    Parameters
    ----------
    content
        The categories. Can be a strategy for [`Content`][ak.contents.Content], a
        concrete [`Content`][ak.contents.Content] instance, or `None` to draw from
        `leaf_contents()` with `unique=True`; a draw with more than one `NaT` is
        rejected. The categories must be a 1-D
        [`NumpyArray`][ak.contents.NumpyArray], an
        [`EmptyArray`][ak.contents.EmptyArray], string content, or bytestring
        content, and unique as above. A concrete `content` that is not raises
        `ValueError`; draws from a strategy that are not are rejected.
    allow_indexed
        No [`IndexedArray`][ak.contents.IndexedArray] is generated if `False`.
    allow_indexed_option
        No [`IndexedOptionArray`][ak.contents.IndexedOptionArray] is generated if
        `False`.
    min_size
        Lower bound on the index length, i.e., `len(result)`.
    max_size
        Upper bound on the index length, i.e., `len(result)`. If `None`, twice the
        content length is used.

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
