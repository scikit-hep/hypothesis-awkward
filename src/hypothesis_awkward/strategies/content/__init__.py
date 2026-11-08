"""Strategies for contents that do not correspond to a single Content class.

The strategies for [`Content`][ak.contents.Content] and its subclasses are in
`hypothesis_awkward.strategies`.
"""

__all__ = [
    'leaf_contents',
    'option_contents',
]

from .leaf import leaf_contents
from .option import option_contents
