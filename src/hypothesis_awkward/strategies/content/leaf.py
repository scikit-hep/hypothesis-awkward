import numpy as np
from hypothesis import strategies as st

from awkward.contents import EmptyArray, ListOffsetArray, NumpyArray, RecordArray
from hypothesis_awkward import strategies as st_ak


def leaf_contents(
    *,
    dtypes: st.SearchStrategy[np.dtype] | None = None,
    allow_nan: bool = True,
    min_size: int = 0,
    max_size: int = 10,
    unique: bool = False,
    allow_numpy: bool = True,
    allow_empty: bool = True,
    allow_string: bool = True,
    allow_bytestring: bool = True,
    allow_zero_field_record: bool = True,
) -> st.SearchStrategy[NumpyArray | EmptyArray | ListOffsetArray | RecordArray]:
    """Strategy for leaf content types.

    This strategy generates [`EmptyArray`][ak.contents.EmptyArray], a zero-field
    [`RecordArray`][ak.contents.RecordArray], bytestring content, string content, and
    [`NumpyArray`][ak.contents.NumpyArray] and shrinks in that order towards
    [`EmptyArray`][ak.contents.EmptyArray].

    Parameters
    ----------
    dtypes
        A strategy for NumPy scalar dtypes used in
        [`NumpyArray`][ak.contents.NumpyArray]. If `None`, the default strategy that
        generates any scalar dtype supported by Awkward Array is used. Does not affect
        string or bytestring content.
    allow_nan
        No `NaN`/`NaT` values are generated in [`NumpyArray`][ak.contents.NumpyArray]
        if `False`.
    min_size
        Minimum number of elements.
    max_size
        Maximum number of elements.
    unique
        No two elements are equal if `True`. Passed on to the
        [`NumpyArray`][ak.contents.NumpyArray], string, and bytestring strategies. A
        zero-field [`RecordArray`][ak.contents.RecordArray], whose elements are all
        equal, is then at most one element long and is not generated if
        `min_size > 1`. With a given `dtypes`, a drawn dtype that cannot supply
        `min_size` distinct values is rejected; if every drawn dtype is rejected, no
        [`NumpyArray`][ak.contents.NumpyArray] is generated, and Hypothesis raises
        `Unsatisfiable` if no other leaf type is allowed.
    allow_numpy
        No [`NumpyArray`][ak.contents.NumpyArray] is generated if `False`.
    allow_empty
        No [`EmptyArray`][ak.contents.EmptyArray] is generated if `False`.
    allow_string
        No string content is generated if `False`.
    allow_bytestring
        No bytestring content is generated if `False`.
    allow_zero_field_record
        No zero-field [`RecordArray`][ak.contents.RecordArray] is generated if `False`.
        Its length is drawn between `min_size` and `max_size`, since a zero-field record
        stores no data and nothing derives its length.

    Raises
    ------
    ValueError
        If no content types are possible with the given options.

    Returns
    -------
    NumpyArray | EmptyArray | ListOffsetArray | RecordArray
    """
    options = list[
        st.SearchStrategy[NumpyArray | EmptyArray | ListOffsetArray | RecordArray]
    ]()

    # Append strategies in optimal order for shrinking.
    if allow_empty and min_size <= 0 <= max_size:
        options.append(st_ak.empty_array_contents())
    if allow_zero_field_record and not (unique and min_size > 1):
        # All elements of a zero-field record are equal; under `unique`, at most one.
        max_length = min(max_size, 1) if unique else max_size
        s = st_ak.record_array_contents(
            max_fields=0, min_length=min_size, max_length=max_length
        )
        options.append(s)
    if allow_bytestring:
        s = st_ak.bytestring_contents(
            min_size=min_size, max_size=max_size, unique=unique
        )
        options.append(s)
    if allow_string:
        s = st_ak.string_contents(min_size=min_size, max_size=max_size, unique=unique)
        options.append(s)
    if allow_numpy:
        s = st_ak.numpy_array_contents(
            dtypes=dtypes,
            allow_nan=allow_nan,
            min_size=min_size,
            max_size=max_size,
            unique=unique,
        )
        options.append(s)

    if not options:
        raise ValueError('no content types are possible with the given options')

    return st.one_of(options)
