import numpy as np
from hypothesis import strategies as st

# In order of the ratio to `us` either way, larger unit first on ties
SORTED_BASE_UNITS = tuple('us ms ns s ps m fs h D W as M Y'.split())


def datetime_units(*, allow_multiples: bool = True) -> st.SearchStrategy[str]:
    """Strategy for units of NumPy datetime and timedelta dtypes.

    A drawn unit, e.g., `'m'` or `'10m'`, makes a dtype name such as `datetime64[m]` or
    `timedelta64[10m]`.

    Parameters
    ----------
    allow_multiples
        Draw multiples of units, e.g., `'10m'`, as well as units, e.g., `'m'`, if
        `True`, else only units.
    """
    base_units = st.sampled_from(SORTED_BASE_UNITS)
    if not allow_multiples:
        return base_units
    # TODO: Restore `max_value=2**31 - 1` when Awkward Array fixes the cache.
    # Up to 999, no two units make equal dtypes with different names, e.g.,
    # `1000ps` and `ns`. Such dtypes share or corrupt the entries of the cache of
    # Awkward Array's `dtype_to_primitive()`.
    # https://github.com/scikit-hep/awkward/issues/4581
    multipliers = st.integers(min_value=1, max_value=999)
    multiples = st.builds(lambda n, u: f'{n}{u}', multipliers, base_units)
    return st.one_of(base_units, multiples)


def _distance_from_us(unit: str) -> tuple[float, float]:
    """Sort key for `SORTED_BASE_UNITS`."""
    length = np.array(1, f'M8[{unit}]') - np.array(0, 'M8[us]')  # 'M' is Jan 1970
    ratio = float(length / np.timedelta64(1, 'us'))
    return max(ratio, 1 / ratio), -ratio


assert list(SORTED_BASE_UNITS) == sorted(SORTED_BASE_UNITS, key=_distance_from_us)
