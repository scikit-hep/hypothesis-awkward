import functools
from typing import Protocol

import numpy as np
from hypothesis import assume
from hypothesis import strategies as st

from awkward.contents import Content
from hypothesis_awkward.util.awkward import content_size, leaf_size
from hypothesis_awkward.util.safe import safe_compare as sc

from .bit_masked_array import bit_masked_array_from_contents
from .byte_masked_array import byte_masked_array_from_contents
from .indexed_option_array import indexed_option_array_from_contents
from .leaf import leaf_contents
from .list_array import list_array_from_contents
from .list_offset_array import list_offset_array_from_contents
from .option import StOption, option_from_contents
from .record_array import record_array_from_contents
from .regular_array import regular_array_from_contents
from .union_array import union_array_from_contents
from .unmasked_array import unmasked_array_from_contents


@st.composite
def contents(
    draw: st.DrawFn,
    *,
    dtypes: st.SearchStrategy[np.dtype] | None = None,
    max_size: int = 50,
    allow_nan: bool = True,
    allow_numpy: bool = True,
    allow_empty: bool = True,
    allow_string: bool = True,
    allow_bytestring: bool = True,
    allow_regular: bool = True,
    allow_list_offset: bool = True,
    allow_list: bool = True,
    allow_record: bool = True,
    allow_union: bool = True,
    allow_indexed_option: bool = True,
    allow_byte_masked: bool = True,
    allow_bit_masked: bool = True,
    allow_unmasked: bool = True,
    max_leaf_size: int | None = None,
    max_depth: int | None = None,
    max_length: int | None = None,
    allow_union_root: bool = True,
    allow_option_root: bool = True,
) -> Content:
    """Strategy for Awkward Array content layouts.

    The current implementation generates the following layouts:

    - [`EmptyArray`][ak.contents.EmptyArray]
    - [`NumpyArray`][ak.contents.NumpyArray]
    - [`RegularArray`][ak.contents.RegularArray]
    - [`ListArray`][ak.contents.ListArray]
    - [`ListOffsetArray`][ak.contents.ListOffsetArray]
    - Strings
    - Bytestrings
    - [`RecordArray`][ak.contents.RecordArray]
    - [`IndexedOptionArray`][ak.contents.IndexedOptionArray]
    - [`ByteMaskedArray`][ak.contents.ByteMaskedArray]
    - [`BitMaskedArray`][ak.contents.BitMaskedArray]
    - [`UnmaskedArray`][ak.contents.UnmaskedArray]
    - [`UnionArray`][ak.contents.UnionArray]

    Each type can be excluded separately with the corresponding ``allow_*`` argument.

    The ``max_size`` is the main argument for constraining the array size. It counts most
    of the scalar values in the layout, including data elements, offsets, indices, field
    names, and parameters.  The array size can also be constrained with
    ``max_leaf_size``, ``max_depth``, and ``max_length``.

    Parameters
    ----------
    dtypes
        A strategy for NumPy scalar dtypes used in [`NumpyArray`][ak.contents.NumpyArray]. If ``None``, the
        default strategy that generates any scalar dtype supported by Awkward Array is
        used. Does not affect string or bytestring content.
    max_size
        Upper bound on the number of scalars in the generated content. Counts data
        elements, offsets, indices, field names, and parameters.
    allow_nan
        No ``NaN``/``NaT`` values are generated in [`NumpyArray`][ak.contents.NumpyArray] if ``False``.
    allow_numpy
        No [`NumpyArray`][ak.contents.NumpyArray] is generated if ``False``.
    allow_empty
        No [`EmptyArray`][ak.contents.EmptyArray] is generated if ``False``. [`EmptyArray`][ak.contents.EmptyArray] has Awkward type
        ``unknown`` and carries no data. Unlike [`NumpyArray`][ak.contents.NumpyArray], it is unaffected by
        ``dtypes`` and ``allow_nan``.
    allow_string
        No string content is generated if ``False``. A string is represented as a
        [`ListOffsetArray`][ak.contents.ListOffsetArray] wrapping a ``NumpyArray(uint8)``. Each character (uint8) and
        offset in the [`ListOffsetArray`][ak.contents.ListOffsetArray] counts toward ``max_size``. A string is
        considered a single leaf element in counting toward ``max_leaf_size`` and
        ``max_depth``.  Each string (not character) counts toward ``max_leaf_size``. A
        string does not count toward ``max_depth``. Unaffected by ``dtypes`` and
        ``allow_nan``.
    allow_bytestring
        No bytestring content is generated if ``False``. A bytestring is represented as a
        [`ListOffsetArray`][ak.contents.ListOffsetArray] wrapping a ``NumpyArray(uint8)``. Each byte (uint8) and
        offset in the [`ListOffsetArray`][ak.contents.ListOffsetArray] counts toward ``max_size``. A bytestring is
        considered a single leaf element in counting toward ``max_leaf_size`` and
        ``max_depth``. Each bytestring (not byte) counts toward ``max_leaf_size``. A
        bytestring does not count toward ``max_depth``. Unaffected by ``dtypes`` and
        ``allow_nan``.
    allow_regular
        No [`RegularArray`][ak.contents.RegularArray] is generated if ``False``.
    allow_list_offset
        No [`ListOffsetArray`][ak.contents.ListOffsetArray] is generated if ``False``.
    allow_list
        No [`ListArray`][ak.contents.ListArray] is generated if ``False``.
    allow_record
        No [`RecordArray`][ak.contents.RecordArray] is generated if ``False``.
    allow_union
        No [`UnionArray`][ak.contents.UnionArray] is generated if ``False``.
    allow_indexed_option
        No [`IndexedOptionArray`][ak.contents.IndexedOptionArray] is generated if ``False``.
    allow_byte_masked
        No [`ByteMaskedArray`][ak.contents.ByteMaskedArray] is generated if ``False``.
    allow_bit_masked
        No [`BitMaskedArray`][ak.contents.BitMaskedArray] is generated if ``False``.
    allow_unmasked
        No [`UnmaskedArray`][ak.contents.UnmaskedArray] is generated if ``False``.
    max_leaf_size
        Maximum total number of leaf elements in the generated content. Each numerical
        value, including complex and datetime, counts as one. Each string and bytestring
        (not character or byte) counts as one.
    max_depth
        Maximum nesting depth. Each [`RegularArray`][ak.contents.RegularArray], [`ListOffsetArray`][ak.contents.ListOffsetArray], [`ListArray`][ak.contents.ListArray],
        [`RecordArray`][ak.contents.RecordArray], and [`UnionArray`][ak.contents.UnionArray] layer adds one level, excluding those that
        form string or bytestring content. No constraint when ``None`` (the default).
    max_length
        Maximum ``len()`` of the generated array. No constraint when ``None`` (the
        default).
    allow_union_root
        The outermost content node cannot be a [`UnionArray`][ak.contents.UnionArray] if ``False``. Unlike
        ``allow_union``, this does not prevent [`UnionArray`][ak.contents.UnionArray] at deeper levels. Awkward
        Array does not allow a [`UnionArray`][ak.contents.UnionArray] to directly contain another [`UnionArray`][ak.contents.UnionArray].
    allow_option_root
        The outermost content node cannot be an option type if ``False``. Does not
        affect deeper levels. Prevents option-inside-option nesting.

    Returns
    -------
    Content

    Examples
    --------
    >>> c = contents().example()
    >>> isinstance(c, Content)
    True
    """
    st_leaf = functools.partial(
        leaf_contents,
        dtypes=dtypes,
        allow_nan=allow_nan,
        allow_numpy=allow_numpy,
        allow_empty=allow_empty,
        allow_string=allow_string,
        allow_bytestring=allow_bytestring,
    )
    leaf_max_size = max_size
    if max_leaf_size is not None:
        leaf_max_size = min(leaf_max_size, max_leaf_size)
    if max_length is not None:
        leaf_max_size = min(leaf_max_size, max_length)

    any_wrapper = any(
        (allow_regular, allow_list_offset, allow_list, allow_record, allow_union)
    )
    any_option = (
        any(
            (
                allow_indexed_option,
                allow_byte_masked,
                allow_bit_masked,
                allow_unmasked,
            )
        )
        and allow_option_root
    )
    leaf_only = (
        not any_wrapper and not any_option or max_leaf_size == 0 or max_size == 0
    )

    def _check(c: Content) -> Content:
        assume(content_size(c) <= max_size)
        return c

    if leaf_only:
        return _check(draw(st_leaf(min_size=0, max_size=leaf_max_size)))

    if max_depth is not None and max_depth <= 0 or not draw(st.booleans()):
        return _check(draw(st_leaf(min_size=0, max_size=leaf_max_size)))

    recurse = functools.partial(
        contents,
        dtypes=dtypes,
        allow_nan=allow_nan,
        allow_numpy=allow_numpy,
        allow_empty=allow_empty,
        allow_string=allow_string,
        allow_bytestring=allow_bytestring,
        max_depth=None if max_depth is None else max_depth - 1,
        allow_regular=allow_regular,
        allow_list_offset=allow_list_offset,
        allow_list=allow_list,
        allow_record=allow_record,
        allow_union=allow_union,
        allow_indexed_option=allow_indexed_option,
        allow_byte_masked=allow_byte_masked,
        allow_bit_masked=allow_bit_masked,
        allow_unmasked=allow_unmasked,
    )

    # Choose wrapper type from allow_* flags
    candidates = list[_StFromContents]()
    if allow_regular:
        candidates.append(regular_array_from_contents)
    if allow_list_offset:
        candidates.append(list_offset_array_from_contents)
    if allow_list:
        candidates.append(list_array_from_contents)
    if allow_record:
        candidates.append(record_array_from_contents)
    if allow_union and allow_union_root:
        candidates.append(union_array_from_contents)
    if allow_indexed_option and allow_option_root:
        candidates.append(indexed_option_array_from_contents)
    if allow_byte_masked and allow_option_root:
        candidates.append(byte_masked_array_from_contents)
    if allow_bit_masked and allow_option_root:
        candidates.append(bit_masked_array_from_contents)
    if allow_unmasked and allow_option_root:
        candidates.append(unmasked_array_from_contents)

    if not candidates:
        return _check(draw(st_leaf(min_size=0, max_size=leaf_max_size)))

    st_option_: StOption | None = (
        functools.partial(
            option_from_contents,
            allow_indexed_option=allow_indexed_option,
            allow_byte_masked=allow_byte_masked,
            allow_bit_masked=allow_bit_masked,
            allow_unmasked=allow_unmasked,
        )
        if any_option
        else None
    )

    st_wrapper = draw(st.sampled_from(candidates))
    return draw(
        st_wrapper(
            recurse,
            max_size=max_size,
            max_leaf_size=max_leaf_size,
            max_length=max_length,
            st_option=st_option_,
        )
    )


class StContent(Protocol):
    def __call__(
        self,
        *,
        max_size: int,
        max_leaf_size: int | None,
        allow_union_root: bool = ...,
        allow_option_root: bool = ...,
    ) -> st.SearchStrategy[Content]: ...


class _StFromContents(Protocol):
    def __call__(
        self,
        content: StContent,
        *,
        max_size: int,
        max_leaf_size: int | None = ...,
        max_length: int | None = ...,
        st_option: StOption | None = ...,
    ) -> st.SearchStrategy[Content]: ...


@st.composite
def content_lists(
    draw: st.DrawFn,
    st_content: StContent = contents,
    *,
    max_size: int = 50,
    max_leaf_size: int | None = None,
    min_len: int = 0,
    max_len: int | None = None,
    all_option_or_none: bool = False,
    st_option: StOption | None = None,
) -> list[Content]:
    """Strategy for lists of contents within a size budget.

    Parameters
    ----------
    st_content
        A callable that accepts ``max_size`` and ``max_leaf_size`` and returns a strategy
        for a single content.
    max_size
        Upper bound on total ``content_size()`` across all contents in the list.
    max_leaf_size
        Maximum total number of leaf elements across all contents in the list.
    min_len
        Minimum number of contents in the list.
    max_len
        Maximum number of contents in the list. By default there is no upper bound.
    all_option_or_none
        If ``True``, enforce all-or-none option typing: the first child decides
        whether all children are option-wrapped. Requires ``st_option``.
    st_option
        A callable conforming to ``StOption`` that wraps content in an option type.
        Required when ``all_option_or_none`` is ``True``.

    Returns
    -------
    list[Content]
    """
    remaining_leaf = max_leaf_size
    remaining_total = max_size
    contents_ = list[Content]()

    def _remaining_max_leaf_size() -> int | None:
        return max(remaining_leaf, 0) if remaining_leaf is not None else None

    def _draw_content() -> Content:
        return draw(
            st_content(
                max_size=max(remaining_total, 0),
                max_leaf_size=_remaining_max_leaf_size(),
            )
        )

    def _draw_option() -> Content:
        assert st_option is not None
        return draw(
            st_option(
                functools.partial(st_content, allow_option_root=False),
                max_size=max(remaining_total, 0),
                max_leaf_size=_remaining_max_leaf_size(),
            )
        )

    # After the first child, use_option decides whether to wrap all in option
    use_option: bool | None = None  # None = not yet decided

    def _draw() -> Content:
        if use_option is True:
            return _draw_option()
        if use_option is False:
            return _draw_content()
        # Not yet decided (first child) — draw from st_content
        return _draw_content()

    for i in range(min_len):
        c = _draw()
        if all_option_or_none and use_option is None and st_option is not None:
            use_option = c.is_option
            if not use_option:
                # Ensure subsequent draws don't produce option at root
                st_content = functools.partial(st_content, allow_option_root=False)
        if remaining_leaf is not None:
            remaining_leaf -= leaf_size(c)
        remaining_total -= content_size(c)
        contents_.append(c)
    while (
        draw(st.booleans())
        and sc(remaining_leaf) > 0
        and remaining_total > 0
        and len(contents_) < sc(max_len)
    ):
        c = _draw()
        if remaining_leaf is not None:
            remaining_leaf -= leaf_size(c)
        remaining_total -= content_size(c)
        contents_.append(c)
    return contents_
