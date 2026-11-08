import numpy as np
from hypothesis import strategies as st

import awkward as ak
from awkward.contents import ListOffsetArray, NumpyArray


@st.composite
def string_contents(
    draw: st.DrawFn,
    *,
    alphabet: st.SearchStrategy[str] | None = None,
    min_size: int = 0,
    max_size: int = 10,
    unique: bool = False,
) -> ListOffsetArray:
    """Strategy for string [`ak.contents.ListOffsetArray`][] instances.

    Parameters
    ----------
    alphabet
        A strategy for characters used in the generated strings. If `None`, the full
        Unicode range is used.
    min_size
        Minimum number of strings.
    max_size
        Maximum number of strings.
    unique
        No two strings are equal if `True`. Strings are compared by code point, without
        Unicode normalization.

    Examples
    --------
    >>> import awkward as ak
    >>> from hypothesis import given
    >>> from hypothesis import strategies as st
    >>> import hypothesis_awkward.strategies as st_ak

    Restrict the characters:

    >>> @given(c=st_ak.constructors.string_contents(alphabet=st.sampled_from('abc')))
    ... def test_alphabet(c):
    ...     assert all(set(s) <= set('abc') for s in ak.to_list(c))
    >>> test_alphabet()

    Draw two to five distinct strings:

    >>> @given(
    ...     c=st_ak.constructors.string_contents(min_size=2, max_size=5, unique=True)
    ... )
    ... def test_unique(c):
    ...     strings = ak.to_list(c)
    ...     assert 2 <= len(strings) <= 5
    ...     assert len(set(strings)) == len(strings)
    >>> test_unique()
    """
    text_st = st.text() if alphabet is None else st.text(alphabet=alphabet)
    strings = draw(
        st.lists(text_st, min_size=min_size, max_size=max_size, unique=unique)
    )
    encoded = [s.encode('utf-8') for s in strings]
    offsets = np.zeros(len(encoded) + 1, dtype=np.int64)
    for i, b in enumerate(encoded):
        offsets[i + 1] = offsets[i] + len(b)
    data = b''.join(encoded)
    content = NumpyArray(
        np.frombuffer(data, dtype=np.uint8) if data else np.array([], dtype=np.uint8),
        parameters={'__array__': 'char'},
    )
    return ListOffsetArray(
        ak.index.Index64(offsets), content, parameters={'__array__': 'string'}
    )
