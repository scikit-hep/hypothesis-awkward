import numpy as np
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak


def st_arrays(
    dtype: np.dtype | st.SearchStrategy[np.dtype] | None = None,
    allow_nan: bool = True,
    max_size: int = 10,
) -> st.SearchStrategy[ak.Array]:
    '''Tentative strategy for Awkward Arrays (combines from_numpy and from_list).'''
    return st.one_of(
        st_ak.from_numpy(dtype=dtype, allow_nan=allow_nan, max_size=max_size),
        st_ak.from_list(dtype=dtype, allow_nan=allow_nan, max_size=max_size),
    )
