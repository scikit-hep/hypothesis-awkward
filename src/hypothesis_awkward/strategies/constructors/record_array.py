import string

from hypothesis import assume
from hypothesis import strategies as st

from awkward.contents import Content, RecordArray
from hypothesis_awkward._util import content_size

from ._types import StContent, StOption

_MAX_LENGTH_NO_FIELDS = 5


@st.composite
def record_array_contents(
    draw: st.DrawFn,
    contents: list[Content] | st.SearchStrategy[list[Content]] | None = None,
    *,
    max_fields: int = 5,
    allow_tuple: bool = True,
    min_length: int = 0,
    max_length: int | None = None,
) -> RecordArray:
    """Strategy for [`ak.contents.RecordArray`][] instances.

    Parameters
    ----------
    contents
        Child contents. Can be a strategy for a list of [`Content`][ak.contents.Content],
        a concrete list, or `None` to draw random children.
    max_fields
        Maximum number of fields when `contents` is `None`.
    allow_tuple
        Allow tuple records (no field names) if `True`.
    min_length
        Lower bound on the record length, i.e., `len(result)`.
    max_length
        Upper bound on the record length, i.e., `len(result)`.

    Notes
    -----
    With at least one field, the length is derived from the children as
    `min(len(c) for c in contents)`, capped by `max_length`. A record with no fields has
    no children and stores no data, so nothing derives its length; it is drawn between
    `min_length` and `max_length` instead. When `max_length` is `None`, the ceiling is an
    internal value, raised to `min_length` if that is larger.

    Examples
    --------
    >>> from hypothesis import given
    >>> import hypothesis_awkward.strategies as st_ak

    Limit the number of fields and the record length:

    >>> @given(
    ...     c=st_ak.constructors.record_array_contents(
    ...         max_fields=3, min_length=2, max_length=4
    ...     )
    ... )
    ... def test_options(c):
    ...     assert len(c.fields) <= 3
    ...     assert 2 <= len(c) <= 4
    >>> test_options()

    Exclude tuples, i.e., records without field names:

    >>> @given(c=st_ak.constructors.record_array_contents(allow_tuple=False))
    ... def test_named(c):
    ...     assert not c.is_tuple
    >>> test_named()

    A record with no fields gets its length drawn, since no field content determines
    it:

    >>> @given(
    ...     c=st_ak.constructors.record_array_contents(
    ...         max_fields=0, min_length=3, max_length=3
    ...     )
    ... )
    ... def test_zero_fields(c):
    ...     assert len(c.contents) == 0
    ...     assert len(c) == 3
    >>> test_zero_fields()
    """
    # Imported here because content.py imports this module.
    from .content import content_lists

    match contents:
        case None:
            contents = draw(content_lists(max_len=max_fields))
        case st.SearchStrategy():
            contents = draw(contents)
        case list():
            pass
    assert isinstance(contents, list)

    if allow_tuple:
        is_tuple = draw(st.booleans())
    else:
        is_tuple = False

    if is_tuple:
        fields = None
    else:
        st_names = st.text(alphabet=string.ascii_letters, max_size=3)
        fields = draw(
            st.lists(
                st_names,
                min_size=len(contents),
                max_size=len(contents),
                unique=True,
            )
        )

    if contents:
        candidate_length = min(len(c) for c in contents)
        assume(candidate_length >= min_length)
        length = None if max_length is None else min(candidate_length, max_length)
    else:
        # A record with no fields stores no data, so nothing derives its length.
        if max_length is None:
            max_length = max(min_length, _MAX_LENGTH_NO_FIELDS)
        assume(min_length <= max_length)
        length = draw(st.integers(min_value=min_length, max_value=max_length))
    return RecordArray(contents, fields=fields, length=length)


@st.composite
def record_array_from_contents(
    draw: st.DrawFn,
    content: StContent,
    *,
    max_size: int,
    max_leaf_size: int | None = None,
    min_fields: int = 1,
    max_fields: int | None = None,
    min_length: int = 0,
    max_length: int | None = None,
    st_option: StOption | None = None,
) -> RecordArray:
    """Strategy for [`ak.contents.RecordArray`][] instances within a size budget.

    Draws at least `min_fields` and at most `max_fields` children via
    `content_lists()`, then wraps them in a
    [`RecordArray`][ak.contents.RecordArray] with generated or omitted field names.

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
    min_fields
        Lower bound on the number of fields, i.e., `len(result.contents)`.
    max_fields
        Upper bound on the number of fields, i.e., `len(result.contents)`.
        Unbounded if `None`.
    min_length
        Lower bound on `len(result)`.
    max_length
        Upper bound on `len(result)`. Unbounded if `None`.
    st_option
        Accepted for `_StFromContents` compatibility; unused in this variant.

    Examples
    --------
    >>> from hypothesis_awkward.strategies.constructors import contents
    >>> c = record_array_from_contents(
    ...     contents, max_size=20, max_leaf_size=10, min_length=2, max_length=5
    ... ).example()
    >>> isinstance(c, RecordArray)
    True

    >>> 2 <= len(c) <= 5
    True

    Limit the number of fields:

    >>> c = record_array_from_contents(
    ...     contents, max_size=20, min_fields=2, max_fields=3
    ... ).example()
    >>> 2 <= len(c.contents) <= 3
    True
    """
    # Imported here because content.py imports this module.
    from .content import content_lists

    children = draw(
        content_lists(
            content,
            max_size=max_size,
            max_leaf_size=max_leaf_size,
            min_len=min_fields,
            max_len=max_fields,
        )
    )
    result = draw(
        record_array_contents(children, min_length=min_length, max_length=max_length)
    )
    assume(content_size(result) <= max_size)
    return result
