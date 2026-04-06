import math
from typing import Any

import numpy as np
from hypothesis import Phase, find, given, settings
from hypothesis import strategies as st

import hypothesis_awkward.strategies as st_ak


def _is_nan(item: Any) -> bool:
    if item is None:
        # NaT becomes None
        return True
    elif isinstance(item, complex):
        return math.isnan(item.real) or math.isnan(item.imag)
    elif isinstance(item, float):
        return math.isnan(item)
    return False


@given(data=st.data())
def test_items_from_dtype(data: st.DataObject) -> None:
    dtype = data.draw(st_ak.builtin_safe_dtypes(), label='dtype')
    allow_nan = data.draw(st.booleans(), label='allow_nan')
    item = data.draw(st_ak.items_from_dtype(dtype, allow_nan=allow_nan), label='item')

    assert type(item).__module__ in ('builtins', 'datetime')

    if not allow_nan:
        assert not _is_nan(item)

    def _to_dtype_to_item(dtype: np.dtype, item: Any) -> Any:
        if dtype.kind == 'M':  # datetime64
            assert not isinstance(item, int)
            # datetime64 requires the unit.
            unit, _ = np.datetime_data(dtype)
            n = dtype.type(item, unit)
        else:
            n = dtype.type(item)
        return n.item()

    result = _to_dtype_to_item(dtype, item)
    if _is_nan(item):
        assert _is_nan(result)
    else:
        assert result == item


def test_builtin_safe_dtype_names_shrinks_to_bool() -> None:
    """Assert that builtin_safe_dtype_names() shrinks to bool."""
    result = find(
        st_ak.builtin_safe_dtype_names(),
        lambda _: True,
        settings=settings(database=None),
    )
    assert result == 'bool'


def test_builtin_safe_dtypes_shrinks_to_bool() -> None:
    """Assert that builtin_safe_dtypes() shrinks to bool."""
    result = find(
        st_ak.builtin_safe_dtypes(),
        lambda _: True,
        settings=settings(database=None),
    )
    assert result == np.dtype('bool')


def test_draw_nan() -> None:
    """Assert that NaN can be drawn by default."""
    find(
        st_ak.items_from_dtype(np.dtype('float64')),
        lambda item: isinstance(item, float) and math.isnan(item),
        settings=settings(phases=[Phase.generate], max_examples=2000),
    )


def test_draw_nat() -> None:
    """Assert that NaT can be drawn by default."""
    find(
        st_ak.items_from_dtype(np.dtype('datetime64[us]')),
        lambda item: item is None,
        settings=settings(phases=[Phase.generate]),
    )
