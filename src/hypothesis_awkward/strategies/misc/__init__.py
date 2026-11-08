"""Helper strategies that are not specific to Awkward Array.

They help draw optional arguments in tests: `none_or()` for an argument that can be
`None`, and `ranges()` for a pair of minimum and maximum values.
"""

__all__ = [
    'none_or',
    'ranges',
]

from .idiom import none_or
from .range import ranges
