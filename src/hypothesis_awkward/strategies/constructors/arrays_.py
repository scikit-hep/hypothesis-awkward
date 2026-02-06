import math

import numpy as np
from hypothesis import strategies as st

import awkward as ak
from hypothesis_awkward.strategies.numpy import numpy_arrays

MAX_REGULAR_SIZE = 5


@st.composite
def arrays(
    draw: st.DrawFn,
    dtypes: st.SearchStrategy[np.dtype] | None = None,
    allow_nan: bool = False,
    allow_regular: bool = True,
    max_size: int = 10,
    max_depth: int = 3,
) -> ak.Array:
    '''Strategy for Awkward Arrays built from direct Content constructors.

    Parameters
    ----------
    dtypes
        A strategy for NumPy dtypes used in leaf ``NumpyArray`` nodes.
        If ``None``, uses ``supported_dtypes()``.
    allow_nan
        Generate potentially ``NaN``/``NaT`` values for relevant dtypes
        if ``True``.
    allow_regular
        Allow wrapping the leaf ``NumpyArray`` in one or more
        ``RegularArray`` layers if ``True``.
    max_size
        Maximum total number of leaf scalars in the generated array
        (i.e., the sum of ``arr.size`` across all leaf ``NumpyArray``
        nodes).
    max_depth
        Maximum number of nested ``RegularArray`` layers wrapping the
        leaf ``NumpyArray``.  Only effective when *allow_regular* is
        ``True``.

    Examples
    --------
    >>> arrays().example()
    <Array ... type='...'>

    '''
    effective_max_depth = max_depth if allow_regular else 0
    depth = draw(st.integers(min_value=0, max_value=effective_max_depth))

    if depth == 0:
        data = draw(
            numpy_arrays(
                dtype=dtypes,
                allow_structured=False,
                allow_nan=allow_nan,
                max_dims=1,
                max_size=max_size,
            )
        )
        layout: ak.contents.Content = ak.contents.NumpyArray(data)
    else:
        sizes = draw(
            st.lists(
                st.integers(min_value=0, max_value=MAX_REGULAR_SIZE),
                min_size=depth,
                max_size=depth,
            )
        )
        size_product = math.prod(sizes)
        if size_product == 0:
            max_outer = max_size
        else:
            max_outer = max_size // size_product
        outer_len = draw(st.integers(min_value=0, max_value=max_outer))

        # Compute layer lengths top-down
        layer_len = [0] * depth
        layer_len[0] = outer_len
        for i in range(1, depth):
            layer_len[i] = layer_len[i - 1] * sizes[i - 1] if sizes[i - 1] > 0 else 0

        # Compute leaf element count
        leaf_count = layer_len[-1] * sizes[-1] if sizes[-1] > 0 else 0

        data = draw(
            numpy_arrays(
                dtype=dtypes,
                allow_structured=False,
                allow_nan=allow_nan,
                max_dims=1,
                min_size=leaf_count,
                max_size=leaf_count,
            )
        )
        layout = ak.contents.NumpyArray(data)

        # Build RegularArray layers bottom-up
        for i in reversed(range(depth)):
            if sizes[i] > 0:
                layout = ak.contents.RegularArray(layout, size=sizes[i])
            else:
                layout = ak.contents.RegularArray(
                    layout, size=0, zeros_length=layer_len[i]
                )

    return ak.Array(layout)
