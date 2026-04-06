"""The main strategy `arrays()`.

The function `arrays()` is the main strategy of this package. It generates Awkward
Arrays with multiple options to control the layout, data types, missing values, masks,
and other array attributes.
"""

__all__ = ['arrays']

from .array_ import arrays
