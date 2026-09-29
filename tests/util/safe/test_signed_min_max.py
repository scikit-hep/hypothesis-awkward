import itertools

import pytest
from hypothesis import given
from hypothesis import strategies as st

from hypothesis_awkward.util.safe import _signed_max, _signed_min

st_integers = st.integers()
st_floats = st.floats(allow_nan=False)


@given(st.data())
def test_signed_min(data: st.DataObject) -> None:
    st_ = data.draw(st.sampled_from([st_integers, st_floats]))
    vals = data.draw(st.lists(st_))
    default_val = data.draw(st.one_of(st_, st.none()))

    result = _signed_min(vals, default=default_val)

    if vals:
        assert result == min(vals)
        if result == 0:
            result_repr = repr(result)
            zeros_repr = set(repr(v) for v in vals if v == 0)  # `0`, `-0.0`, `0.0`
            assert result_repr in zeros_repr
            if not result_repr == '-0.0':
                assert '-0.0' not in zeros_repr
    else:
        assert result == default_val


@pytest.mark.parametrize(
    'vals',
    [
        *itertools.permutations([0.0, -0.0]),
        *itertools.permutations([0.0, 0, -0.0]),
    ],
)
def test_signed_min_negative_zero(vals: tuple[float, ...]) -> None:
    """Assert that `-0.0` is returned over `0.0` regardless of the order."""
    assert repr(_signed_min(vals)) == '-0.0'


@given(st.data())
def test_signed_max(data: st.DataObject) -> None:
    st_ = data.draw(st.sampled_from([st_integers, st_floats]))
    vals = data.draw(st.lists(st_))
    default_val = data.draw(st.one_of(st_, st.none()))

    result = _signed_max(vals, default=default_val)

    if vals:
        assert result == max(vals)
        if result == 0:
            result_repr = repr(result)
            zeros_repr = set(repr(v) for v in vals if v == 0)  # `0`, `-0.0`, `0.0`
            assert result_repr in zeros_repr
            if result_repr == '-0.0':
                assert zeros_repr == {'-0.0'}
    else:
        assert result == default_val


@pytest.mark.parametrize(
    'vals',
    [
        *itertools.permutations([0.0, -0.0]),
        *itertools.permutations([0.0, 0, -0.0]),
    ],
)
def test_signed_max_non_negative_zero(vals: tuple[float, ...]) -> None:
    """Assert that `0.0` or `0` is returned over `-0.0` regardless of the order."""
    assert repr(_signed_max(vals)) != '-0.0'
