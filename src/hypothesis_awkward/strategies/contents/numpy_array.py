import numpy as np
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak


def numpy_array_contents(
    dtypes: st.SearchStrategy[np.dtype] | None,
    allow_nan: bool,
    max_size: int,
) -> st.SearchStrategy[ak.contents.NumpyArray]:
    '''Strategy for NumpyArray content.'''
    return st_ak.numpy_arrays(
        dtype=dtypes,
        allow_structured=False,
        allow_nan=allow_nan,
        max_dims=1,
        max_size=max_size,
    ).map(ak.contents.NumpyArray)


def counted_numpy_array_contents(
    dtypes: st.SearchStrategy[np.dtype] | None,
    allow_nan: bool,
    max_size: int,
) -> st.SearchStrategy[ak.contents.NumpyArray]:
    '''Strategy for NumpyArray content with a depleting element count.

    Each draw reduces the remaining element count by the length of
    the drawn content. Raises ``NumpyArrayContentCountExhausted``
    when the count reaches zero.
    '''
    remaining = max_size

    @st.composite
    def _contents(draw: st.DrawFn) -> ak.contents.NumpyArray:
        nonlocal remaining
        if remaining == 0:
            raise NumpyArrayContentCountExhausted
        result = draw(numpy_array_contents(dtypes, allow_nan, remaining))
        remaining -= len(result)
        return result

    return _contents()


class NumpyArrayContentCountExhausted(Exception):
    pass
