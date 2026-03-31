import warnings

import numpy as np
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak


@st.composite
def arrays(
    draw: st.DrawFn,
    *,
    dtypes: st.SearchStrategy[np.dtype] | None = None,
    max_size: int = 50,
    allow_nan: bool = True,
    allow_numpy: bool = True,
    allow_empty: bool = True,
    allow_string: bool = True,
    allow_bytestring: bool = True,
    allow_regular: bool = True,
    allow_list_offset: bool = True,
    allow_list: bool = True,
    allow_record: bool = True,
    allow_union: bool = True,
    allow_indexed_option: bool = True,
    allow_byte_masked: bool = True,
    allow_bit_masked: bool = True,
    allow_unmasked: bool = True,
    max_leaf_size: int | None = None,
    max_depth: int = 5,
    max_length: int | None = None,
    allow_virtual: bool = True,
) -> ak.Array:
    '''Strategy for Awkward Arrays.

    This is the main strategy in this package. It is under development. The aim is to
    generate fully general Awkward Arrays, with many options to control layout, data
    types, missing values, masks, and other array attributes.

    In constructing arrays, this strategy follows the Awkward Array User Guide section
    ["Direct constructors"][ak-user-guide-constructors]. It constructs _layouts_ and
    layout types wraps them in an ``ak.Array``. The layouts are instances of subclasses
    of ``ak.contents.Content``.

    [ak-user-guide-constructors]: https://awkward-array.org/doc/stable/user-guide/how-to-create-constructors.html

    By default, when called with no arguments, ``arrays()`` generates the most general
    arrays currently implemented, subject to a finite maximum size. Arguments can be
    provided to exclude certain layouts or data types, or to constrain values and sizes.

    The current implementation generates arrays with the following layouts:

    - ``EmptyArray``
    - ``NumpyArray``
    - ``RegularArray``
    - ``ListArray``
    - ``ListOffsetArray``
    - Strings
    - Bytestrings
    - ``RecordArray``
    - ``IndexedOptionArray``
    - ``ByteMaskedArray``
    - ``BitMaskedArray``
    - ``UnmaskedArray``
    - ``UnionArray``

    Each type can be excluded separately with the corresponding ``allow_*`` argument.

    The ``max_size`` is the main argument for constraining the array size. It counts most
    of the scalar values in the layout, including data elements, offsets, indices, field
    names, and parameters.  The array size can also be constrained with
    ``max_leaf_size``, ``max_depth``, and ``max_length``.

    The ``arrays()`` randomly generates virtual arrays by lazifying buffers. The
    ``allow_virtual`` can be used to disable virtual arrays.

    Parameters
    ----------
    dtypes
        A strategy for NumPy scalar dtypes used in ``NumpyArray``. If ``None``, the
        default strategy that generates any scalar dtype supported by Awkward Array is
        used. Does not affect string or bytestring content.
    max_size
        Upper bound on ``content_size()`` of the generated content. Counts all scalars
        stored in the content tree: data elements, offset/index buffer elements, and
        metadata values (``RegularArray.size``, ``RecordArray`` field names).
    allow_nan
        No ``NaN``/``NaT`` values are generated in ``NumpyArray`` if ``False``.
    allow_numpy
        No ``NumpyArray`` is generated if ``False``.
    allow_empty
        No ``EmptyArray`` is generated if ``False``. ``EmptyArray`` has Awkward type
        ``unknown`` and carries no data. Unlike ``NumpyArray``, it is unaffected by
        ``dtypes`` and ``allow_nan``.
    allow_string
        No string content is generated if ``False``. Strings are represented as a
        ``ListOffsetArray`` wrapping a ``NumpyArray(uint8)``. Each string (not character)
        counts toward ``max_leaf_size``. The string itself does not count toward
        ``max_depth``. Unaffected by ``dtypes`` and ``allow_nan``.
    allow_bytestring
        No bytestring content is generated if ``False``. Bytestrings are represented as a
        ``ListOffsetArray`` wrapping a ``NumpyArray(uint8)``. Each bytestring (not byte)
        counts toward ``max_leaf_size``. The bytestring itself does not count toward
        ``max_depth``. Unaffected by ``dtypes`` and ``allow_nan``.
    allow_regular
        No ``RegularArray`` is generated if ``False``.
    allow_list_offset
        No ``ListOffsetArray`` is generated if ``False``.
    allow_list
        No ``ListArray`` is generated if ``False``.
    allow_record
        No ``RecordArray`` is generated if ``False``.
    allow_union
        No ``UnionArray`` is generated if ``False``.
    allow_indexed_option
        No ``IndexedOptionArray`` is generated if ``False``.
    allow_byte_masked
        No ``ByteMaskedArray`` is generated if ``False``.
    allow_bit_masked
        No ``BitMaskedArray`` is generated if ``False``.
    allow_unmasked
        No ``UnmaskedArray`` is generated if ``False``.
    max_leaf_size
        Maximum total number of leaf elements in the generated content. Each numerical
        value, including complex and datetime, counts as one. Each string and bytestring
        (not character or byte) counts as one.
    max_depth
        Maximum nesting depth. Each RegularArray, ListOffsetArray, ListArray,
        RecordArray, and UnionArray layer adds one level, excluding those that form
        string or bytestring content.
    max_length
        Maximum ``len()`` of the generated array. No constraint when ``None`` (the
        default).
    allow_virtual
        No virtual arrays are generated if ``False``.

    Examples
    --------
    >>> arrays().example()
    <Array ... type='...'>

    '''
    layout = draw(
        st_ak.contents.contents(
            dtypes=dtypes,
            max_size=max_size,
            max_leaf_size=max_leaf_size,
            allow_nan=allow_nan,
            allow_numpy=allow_numpy,
            allow_empty=allow_empty,
            allow_regular=allow_regular,
            allow_list_offset=allow_list_offset,
            allow_list=allow_list,
            allow_string=allow_string,
            allow_bytestring=allow_bytestring,
            allow_record=allow_record,
            allow_union=allow_union,
            allow_indexed_option=allow_indexed_option,
            allow_byte_masked=allow_byte_masked,
            allow_bit_masked=allow_bit_masked,
            allow_unmasked=allow_unmasked,
            max_depth=max_depth,
            max_length=max_length,
        )
    )
    array = ak.Array(layout)
    to_lazify = allow_virtual and draw(st.booleans())
    if not to_lazify:
        return array
    try:
        return _lazify(array)
    except BaseException as e:  # pragma: no cover
        msg = (
            f'An exception {e} was raised while lazifying {array!r}. '
            'Returning the original array.'
        )
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        return array


def _lazify(array: ak.Array) -> ak.Array:
    form, length, buffers = ak.to_buffers(array)
    if not buffers:
        return array
    virtual_buffers = {k: (lambda v=v: v) for k, v in buffers.items()}
    return ak.from_buffers(form, length, virtual_buffers)
