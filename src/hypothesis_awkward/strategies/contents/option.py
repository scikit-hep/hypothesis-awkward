from hypothesis import strategies as st

from awkward.contents import Content

from .bit_masked_array import bit_masked_array_contents
from .byte_masked_array import byte_masked_array_contents
from .indexed_option_array import indexed_option_array_contents
from .unmasked_array import unmasked_array_contents


def option_contents(
    content: st.SearchStrategy[Content] | Content | None = None,
    *,
    max_size: int | None = None,
    allow_indexed_option: bool = True,
    allow_byte_masked: bool = True,
    allow_bit_masked: bool = True,
    allow_unmasked: bool = True,
) -> st.SearchStrategy[Content]:
    '''Strategy for option-type content, selected by ``st.one_of``.

    Picks among ``IndexedOptionArray``, ``ByteMaskedArray``,
    ``BitMaskedArray``, and ``UnmaskedArray``.

    Parameters
    ----------
    content
        Forwarded to each per-type strategy.
    max_size
        Forwarded to ``indexed_option_array_contents()`` to bound the
        index length. ``None`` means no constraint.
    allow_indexed_option
        Include ``IndexedOptionArray`` if ``True``.
    allow_byte_masked
        Include ``ByteMaskedArray`` if ``True``.
    allow_bit_masked
        Include ``BitMaskedArray`` if ``True``.
    allow_unmasked
        Include ``UnmaskedArray`` if ``True``.

    Examples
    --------
    >>> c = option_contents().example()
    >>> isinstance(c, Content)
    True
    '''
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
