import numpy as np
from hypothesis import strategies as st

import awkward as ak
from awkward.contents import ListOffsetArray, NumpyArray


@st.composite
def bytestring_contents(
    draw: st.DrawFn,
    *,
    min_size: int = 0,
    max_size: int = 10,
    unique: bool = False,
) -> ListOffsetArray:
    """Strategy for bytestring [`ak.contents.ListOffsetArray`][] instances.

    Parameters
    ----------
    min_size
        Minimum number of bytestrings.
    max_size
        Maximum number of bytestrings.
    unique
        No two bytestrings are equal if `True`.

    Examples
    --------
    >>> import awkward as ak
    >>> from hypothesis import given
    >>> import hypothesis_awkward.strategies as st_ak

    Draw two to five distinct bytestrings:

    >>> @given(
    ...     c=st_ak.constructors.bytestring_contents(
    ...         min_size=2, max_size=5, unique=True
    ...     )
    ... )
    ... def test_unique(c):
    ...     values = ak.to_list(c)
    ...     assert all(isinstance(v, bytes) for v in values)
    ...     assert 2 <= len(values) <= 5
    ...     assert len(set(values)) == len(values)
    >>> test_unique()
    """
    bytestrings = draw(
        st.lists(st.binary(), min_size=min_size, max_size=max_size, unique=unique)
    )
    offsets = np.zeros(len(bytestrings) + 1, dtype=np.int64)
    for i, b in enumerate(bytestrings):
        offsets[i + 1] = offsets[i] + len(b)
    data = b''.join(bytestrings)
    content = NumpyArray(
        np.frombuffer(data, dtype=np.uint8) if data else np.array([], dtype=np.uint8),
        parameters={'__array__': 'byte'},
    )
    return ListOffsetArray(
        ak.index.Index64(offsets), content, parameters={'__array__': 'bytestring'}
    )
