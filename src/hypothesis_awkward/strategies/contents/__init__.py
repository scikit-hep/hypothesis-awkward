'''Strategies for Contents (layouts).

These strategies are related to the section of Awkward Array User Guide ["Direct
constructors"][ak-user-guide-constructors].

[ak-user-guide-constructors]: https://awkward-array.org/doc/stable/user-guide/how-to-create-constructors.html
'''

__all__ = [
    'contents',
    'content_lists',
    'leaf_contents',
    'empty_array_contents',
    'numpy_array_contents',
    'regular_array_contents',
    'list_array_contents',
    'list_offset_array_contents',
    'string_contents',
    'bytestring_contents',
    'record_array_contents',
    'indexed_option_array_contents',
    'byte_masked_array_contents',
    'bit_masked_array_contents',
    'unmasked_array_contents',
    'union_array_contents',
]

from .bit_masked_array import bit_masked_array_contents
from .byte_masked_array import byte_masked_array_contents
from .bytestring import bytestring_contents
from .content import content_lists, contents
from .empty_array import empty_array_contents
from .indexed_option_array import indexed_option_array_contents
from .leaf import leaf_contents
from .list_array import list_array_contents
from .list_offset_array import list_offset_array_contents
from .numpy_array import numpy_array_contents
from .record_array import record_array_contents
from .regular_array import regular_array_contents
from .string import string_contents
from .union_array import union_array_contents
from .unmasked_array import unmasked_array_contents
