"""Strategies that build Awkward Arrays and their layouts with direct constructors.

`arrays()` is the main strategy of this package. It generates Awkward Arrays, with
options to control the layout, data types, missing values, masks, and other array
attributes.

The other strategies generate layouts: trees of [`Content`][ak.contents.Content] nodes,
which an [`ak.Array`][] wraps. `contents()` generates the layouts that `arrays()`
wraps. Each `*_array_contents()` strategy generates one `Content` subclass;
`leaf_contents()` and `option_contents()` choose among the leaf and the option-type
classes; and `string_contents()`, `bytestring_contents()`, and `categorical_contents()`
generate the layouts of strings, bytestrings, and categorical data.
"""

__all__ = [
    'arrays',
    'contents',
    'empty_array_contents',
    'numpy_array_contents',
    'regular_array_contents',
    'list_offset_array_contents',
    'list_array_contents',
    'record_array_contents',
    'indexed_array_contents',
    'indexed_option_array_contents',
    'byte_masked_array_contents',
    'bit_masked_array_contents',
    'unmasked_array_contents',
    'union_array_contents',
    'string_contents',
    'bytestring_contents',
    'categorical_contents',
    'leaf_contents',
    'option_contents',
]

from .array_ import arrays
from .bit_masked_array import bit_masked_array_contents
from .byte_masked_array import byte_masked_array_contents
from .bytestring import bytestring_contents
from .categorical import categorical_contents
from .content import contents
from .empty_array import empty_array_contents
from .indexed_array import indexed_array_contents
from .indexed_option_array import indexed_option_array_contents
from .leaf import leaf_contents
from .list_array import list_array_contents
from .list_offset_array import list_offset_array_contents
from .numpy_array import numpy_array_contents
from .option import option_contents
from .record_array import record_array_contents
from .regular_array import regular_array_contents
from .string import string_contents
from .union_array import union_array_contents
from .unmasked_array import unmasked_array_contents
