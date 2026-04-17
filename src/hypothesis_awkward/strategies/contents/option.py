from typing import TYPE_CHECKING, Protocol

from hypothesis import strategies as st

from awkward.contents import Content

from .bit_masked_array import bit_masked_array_contents
from .byte_masked_array import byte_masked_array_contents
from .indexed_option_array import indexed_option_array_contents
from .unmasked_array import unmasked_array_contents

if TYPE_CHECKING:
    from .content import StContent


def option_contents(
    content: st.SearchStrategy[Content] | Content | None = None,
    *,
    max_size: int | None = None,
    allow_indexed_option: bool = True,
    allow_byte_masked: bool = True,
    allow_bit_masked: bool = True,
    allow_unmasked: bool = True,
) -> st.SearchStrategy[Content]:
    """Strategy for option-type content, selected by ``st.one_of``.

    Picks among [`IndexedOptionArray`][ak.contents.IndexedOptionArray],
    [`ByteMaskedArray`][ak.contents.ByteMaskedArray],
    [`BitMaskedArray`][ak.contents.BitMaskedArray], and
    [`UnmaskedArray`][ak.contents.UnmaskedArray].

    Parameters
    ----------
    content
        Forwarded to each per-type strategy.
    max_size
        Forwarded to ``indexed_option_array_contents()`` to bound the
        index length. Unbounded if ``None``.
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

    options: list[st.SearchStrategy[Content]] = []
    if allow_indexed_option:
        options.append(indexed_option_array_contents(content, max_size=max_size))
    if allow_byte_masked:
        options.append(byte_masked_array_contents(content))
    if allow_bit_masked:
        options.append(bit_masked_array_contents(content))
    if allow_unmasked:
        options.append(unmasked_array_contents(content))
    return st.one_of(options)


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
            allow_option_root=False,
            allow_union_root=False,
        )
    )
    return draw(
        option_contents(
            child,
            max_size=max_size,
            allow_indexed_option=allow_indexed_option,
            allow_byte_masked=allow_byte_masked,
            allow_bit_masked=allow_bit_masked,
            allow_unmasked=allow_unmasked,
        )
    )
