import numpy as np


def simple_dtype_kinds_in(d: np.dtype) -> set[str]:
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
