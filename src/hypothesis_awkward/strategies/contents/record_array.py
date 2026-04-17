import string
from typing import TYPE_CHECKING

from hypothesis import assume
from hypothesis import strategies as st

from awkward.contents import Content, RecordArray
from hypothesis_awkward import strategies as st_ak
from hypothesis_awkward.util import content_size

if TYPE_CHECKING:
    from .content import StContent
    from .option import StOption


@st.composite
def record_array_contents(
    draw: st.DrawFn,
    contents: list[Content] | st.SearchStrategy[list[Content]] | None = None,
    *,
    max_fields: int = 5,
    allow_tuple: bool = True,
    max_length: int | None = None,
) -> RecordArray:
    """Strategy for [`ak.contents.RecordArray`][] instances.

    Parameters
    ----------
    contents
        Child contents. Can be a strategy for a list of [`Content`][ak.contents.Content], a concrete list, or
        ``None`` to draw random children.
    max_fields
        Maximum number of fields when ``contents`` is ``None``.
    allow_tuple
        Allow tuple records (no field names) if ``True``.
    max_length
        Upper bound on the record length, i.e., ``len(result)``.

    Returns
    -------
    RecordArray

    Examples
    --------
    >>> c = record_array_contents().example()
    >>> isinstance(c, RecordArray)
    True

    Limit the number of fields:

    >>> c = record_array_contents(max_fields=3).example()
    >>> len(c.contents) <= 3
    True

    Limit the record length:

    >>> c = record_array_contents(max_length=4).example()
    >>> len(c) <= 4
    True
    """
    match contents:
        case None:
            contents = draw(st_ak.contents.content_lists(max_len=max_fields))
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

    if not contents:
        length = 0
    elif max_length is not None:
        length = min(min(len(c) for c in contents), max_length)
    else:
        length = None
    return RecordArray(contents, fields=fields, length=length)


@st.composite
def record_array_from_contents(
    draw: st.DrawFn,
    content: 'StContent',
    *,
    max_size: int,
    max_leaf_size: int | None = None,
    max_length: int | None = None,
    st_option: 'StOption | None' = None,
) -> RecordArray:
    """Strategy for [`ak.contents.RecordArray`][] instances within a size budget.

    Draws one or more children via ``content_lists()`` with ``min_len=1``, then wraps
    them in a [`RecordArray`][ak.contents.RecordArray] with generated or omitted field
    names.

    Called by ``contents()`` during recursive tree generation.

    Parameters
    ----------
    content
        A callable that accepts ``max_size`` and ``max_leaf_size`` and returns a strategy
        for a single content.
    max_size
        Upper bound on ``content_size()`` of the result.
    max_leaf_size
        Upper bound on total leaf elements. Unbounded if ``None``.
    max_length
        Upper bound on ``len(result)``. Unbounded if ``None``.

    Returns
    -------
    RecordArray
    """
    children = draw(
        st_ak.contents.content_lists(
            content, max_size=max_size, max_leaf_size=max_leaf_size, min_len=1
        )
    )
    result = draw(record_array_contents(children, max_length=max_length))
    assume(content_size(result) <= max_size)
    return result
