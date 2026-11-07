from hypothesis import strategies as st

from awkward.contents import Content

from .bit_masked_array import bit_masked_array_contents
from .byte_masked_array import byte_masked_array_contents
from .unmasked_array import unmasked_array_contents


def masked_contents(
    content: st.SearchStrategy[Content] | Content | None = None,
    *,
    allow_byte_masked: bool = True,
    allow_bit_masked: bool = True,
    allow_unmasked: bool = True,
) -> st.SearchStrategy[Content]:
    """Strategy for masked content types.

    Picks among [`ByteMaskedArray`][ak.contents.ByteMaskedArray],
    [`BitMaskedArray`][ak.contents.BitMaskedArray], and
    [`UnmaskedArray`][ak.contents.UnmaskedArray]. All three preserve the input
    content's length, so `len(result) == len(content)`.

    Parameters
    ----------
    content
        Forwarded to each per-type strategy.
    allow_byte_masked
        No [`ByteMaskedArray`][ak.contents.ByteMaskedArray] is generated if `False`.
    allow_bit_masked
        No [`BitMaskedArray`][ak.contents.BitMaskedArray] is generated if `False`.
    allow_unmasked
        No [`UnmaskedArray`][ak.contents.UnmaskedArray] is generated if `False`.

    Returns
    -------
    Content

    Examples
    --------
    >>> c = masked_contents().example()
    >>> isinstance(c, Content)
    True
    """
    if not any((allow_byte_masked, allow_bit_masked, allow_unmasked)):
        raise ValueError('at least one masked content type must be allowed')

    # Append in shrink-preferred order (data simplicity): UnmaskedArray has no
    # mask data, ByteMaskedArray uses a straightforward int8 mask, BitMaskedArray
    # uses a bit-packed mask plus extra metadata. `st.one_of` shrinks toward
    # earlier entries.
    options = list[st.SearchStrategy[Content]]()
    if allow_unmasked:
        options.append(unmasked_array_contents(content))
    if allow_byte_masked:
        options.append(byte_masked_array_contents(content))
    if allow_bit_masked:
        options.append(bit_masked_array_contents(content))
    return st.one_of(options)
