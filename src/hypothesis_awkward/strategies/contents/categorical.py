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
    `__array__` set to `'categorical'`. Its content holds the categories, no two of
    which are equal; the index refers to them. The categories are a
    [`NumpyArray`][ak.contents.NumpyArray], an
    [`EmptyArray`][ak.contents.EmptyArray], string content, or bytestring content.

    Parameters
    ----------
    content
        The categories. Can be a strategy for [`Content`][ak.contents.Content], a
        concrete [`Content`][ak.contents.Content] instance, or `None` to draw from
        `leaf_contents()` with `unique=True`. Draws from a strategy that are not
        valid categories are rejected.
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
        concrete `content` is not valid categories.

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
