import numpy as np
from hypothesis import strategies as st

# In order of the ratio to `us` either way, larger unit first on ties
SORTED_BASE_UNITS = tuple('us ms ns s ps m fs h D W as M Y'.split())


def datetime_units() -> st.SearchStrategy[str]:
    """Strategy for units of NumPy datetime and timedelta dtypes.

    A drawn unit, e.g., `'m'`, makes a dtype name such as `datetime64[m]` or
    `timedelta64[m]`.
    """
    return st.sampled_from(SORTED_BASE_UNITS)


def _distance_from_us(unit: str) -> tuple[float, float]:
    """Sort key for `SORTED_BASE_UNITS`."""
    length = np.array(1, f'M8[{unit}]') - np.array(0, 'M8[us]')  # 'M' is Jan 1970
    ratio = float(length / np.timedelta64(1, 'us'))
    return max(ratio, 1 / ratio), -ratio


assert list(SORTED_BASE_UNITS) == sorted(SORTED_BASE_UNITS, key=_distance_from_us)
