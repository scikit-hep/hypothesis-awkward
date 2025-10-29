import numpy as np
from hypothesis import given
from hypothesis import strategies as st

import hypothesis_awkward.strategies as st_ak
from hypothesis_awkward.util import simple_dtype_kinds_in


def _org_imp(d: np.dtype) -> set[str]:
    '''Kinds of simple dtypes (e.g. `i`, `f`, `M`) contained in `d`.'''
    if d.names is None:  # simple dtype
        kind = d.kind
        if kind == 'V' and d.subdtype is not None:
            kind = d.subdtype[0].kind
        return {kind}
    else:  # structured dtype
        kinds = set()
        for name in d.names:
            f = d.fields
            assert f is not None
            kinds.update(simple_dtype_kinds_in(f[name][0]))
        return kinds


@given(data=st.data())
def test_simple_dtype_kinds_in(data: st.DataObject) -> None:
    d = data.draw(st_ak.numpy_dtypes(), label='dtype')
    expected = _org_imp(d)
    actual = simple_dtype_kinds_in(d)
    assert actual == expected
