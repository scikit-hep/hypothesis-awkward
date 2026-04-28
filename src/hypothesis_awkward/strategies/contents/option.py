from typing import TYPE_CHECKING, Protocol

from hypothesis import assume
from hypothesis import strategies as st

from awkward.contents import Content
from hypothesis_awkward import strategies as st_ak
from hypothesis_awkward.util import safe_compare as sc

from .bit_masked_array import bit_masked_array_contents
from .byte_masked_array import byte_masked_array_contents
from .indexed_option_array import indexed_option_array_contents
from .unmasked_array import unmasked_array_contents

if TYPE_CHECKING:
    from .content import StContent


def option_contents(
    content: st.SearchStrategy[Content] | Content | None = None,
    *,
    min_size: int = 0,
    max_size: int | None = None,
    allow_indexed_option: bool = True,
    allow_byte_masked: bool = True,
    allow_bit_masked: bool = True,
    allow_unmasked: bool = True,
) -> st.SearchStrategy[Content]:
    """Strategy for option-type content.

    Picks among [`IndexedOptionArray`][ak.contents.IndexedOptionArray],
    [`ByteMaskedArray`][ak.contents.ByteMaskedArray],
    [`BitMaskedArray`][ak.contents.BitMaskedArray], and
    [`UnmaskedArray`][ak.contents.UnmaskedArray].

    The bounds ``min_size`` and ``max_size`` apply to ``len(result)`` for all
    branches: forwarded as-is for the indexed-option branch and used to filter
    feasible branches for mask-controlled types (whose length equals the
    content's length).

    Parameters
    ----------
    content
        Forwarded to each per-type strategy.
    min_size
        Lower bound on ``len(result)``.
    max_size
        Upper bound on ``len(result)``. Unbounded if ``None``.
    allow_indexed_option
        No [`IndexedOptionArray`][ak.contents.IndexedOptionArray] is generated if
        ``False``.
    allow_byte_masked
        No [`ByteMaskedArray`][ak.contents.ByteMaskedArray] is generated if ``False``.
    allow_bit_masked
        No [`BitMaskedArray`][ak.contents.BitMaskedArray] is generated if ``False``.
    allow_unmasked
        No [`UnmaskedArray`][ak.contents.UnmaskedArray] is generated if ``False``.

    Returns
    -------
    Content

    Examples
    --------
    >>> c = option_contents().example()
    >>> isinstance(c, Content)
    True
    """
    if not any(
        (allow_indexed_option, allow_byte_masked, allow_bit_masked, allow_unmasked)
    ):
        raise ValueError('at least one option content type must be allowed')

    return _option_contents(
        content,
        min_size=min_size,
        max_size=max_size,
        allow_indexed_option=allow_indexed_option,
        allow_byte_masked=allow_byte_masked,
        allow_bit_masked=allow_bit_masked,
        allow_unmasked=allow_unmasked,
    )


@st.composite
def _option_contents(
    draw: st.DrawFn,
    content: st.SearchStrategy[Content] | Content | None,
    *,
    min_size: int,
    max_size: int | None,
    allow_indexed_option: bool,
    allow_byte_masked: bool,
    allow_bit_masked: bool,
    allow_unmasked: bool,
) -> Content:
    """Internal composite that drives ``option_contents``.

    Resolves ``content`` to a concrete instance once so mask-controlled branches
    can be gated on ``len(content)`` without triggering content redraws.
    """
    match content:
        case None:
            content_concrete = draw(
                st_ak.contents.contents(allow_union_root=False, allow_option_root=False)
            )
        case st.SearchStrategy():
            content_concrete = draw(content)
        case _:
            content_concrete = content

    content_len = len(content_concrete)
    mask_in_bounds = min_size <= content_len <= sc(max_size)

    options: list[st.SearchStrategy[Content]] = []
    if allow_indexed_option:
        options.append(
            indexed_option_array_contents(
                content_concrete, min_size=min_size, max_size=max_size
            )
        )
    if mask_in_bounds:
        if allow_byte_masked:
            options.append(byte_masked_array_contents(content_concrete))
        if allow_bit_masked:
            options.append(bit_masked_array_contents(content_concrete))
        if allow_unmasked:
            options.append(unmasked_array_contents(content_concrete))

    assume(options)
    return draw(st.one_of(options))


class StOption(Protocol):
    """Callable that wraps content drawn from a ``StContent`` in an option type."""

    def __call__(
        self,
        content: 'StContent',
        *,
        max_size: int,
        max_leaf_size: int | None,
    ) -> st.SearchStrategy[Content]: ...


@st.composite
def option_from_contents(
    draw: st.DrawFn,
    content: 'StContent',
    *,
    max_size: int,
    max_leaf_size: int | None,
    min_length: int = 0,
    allow_indexed_option: bool = True,
    allow_byte_masked: bool = True,
    allow_bit_masked: bool = True,
    allow_unmasked: bool = True,
) -> Content:
    """Strategy that draws content and wraps it in an option type.

    Conforms to ``StOption`` when partially applied with ``allow_*`` flags.

    Parameters
    ----------
    content
        A callable that accepts ``max_size`` and ``max_leaf_size`` and returns a strategy
        for a single content.
    max_size
        Upper bound on ``content_size()`` of the result.
    max_leaf_size
        Upper bound on total leaf elements. Unbounded if ``None``.
    min_length
        Lower bound on ``len(result)``. Forwarded to the inner ``content(...)`` call
        so the inner content meets the floor, and to ``option_contents`` as
        ``min_size`` for the indexed-option pathway.
    allow_indexed_option
        No [`IndexedOptionArray`][ak.contents.IndexedOptionArray] is generated if
        ``False``.
    allow_byte_masked
        No [`ByteMaskedArray`][ak.contents.ByteMaskedArray] is generated if ``False``.
    allow_bit_masked
        No [`BitMaskedArray`][ak.contents.BitMaskedArray] is generated if ``False``.
    allow_unmasked
        No [`UnmaskedArray`][ak.contents.UnmaskedArray] is generated if ``False``.

    Returns
    -------
    Content
    """
    child = draw(
        content(
            max_size=max_size,
            max_leaf_size=max_leaf_size,
            min_length=min_length,
            allow_option_root=False,
            allow_union_root=False,
        )
    )
    return draw(
        option_contents(
            child,
            min_size=min_length,
            max_size=max_size,
            allow_indexed_option=allow_indexed_option,
            allow_byte_masked=allow_byte_masked,
            allow_bit_masked=allow_bit_masked,
            allow_unmasked=allow_unmasked,
        )
    )
