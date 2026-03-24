'''The main strategy `arrays()`.

The function `arrays()` is the main strategy of this package. It is currently
experimental. The plan is to have `arrays()` generate fully general Awkward Arrays with
multiple options to control the layout, data types, missing values, masks, and other
array attributes.
'''

__all__ = ['arrays']

from .array_ import arrays
