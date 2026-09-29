import itertools

import pytest
from hypothesis import given
from hypothesis import strategies as st

from hypothesis_awkward.util import safe_compare

st_num = st.integers() | st.floats(allow_nan=False) | st.sampled_from([0, 0.0, -0.0])


def test_repr() -> None:
    assert repr(safe_compare(1)) == '1'
    assert repr(safe_compare(None)) == 'GreaterAndLessThanAny()'


@given(st.data())
def test_safe_compare(data: st.DataObject) -> None:
    a = data.draw(st.none() | st_num)
    b = data.draw(st.none() | st_num | st.just(a))

    match a, b:
        case (None, _) | (_, None):
            assert safe_compare(a) <= safe_compare(b)
            assert safe_compare(a) < safe_compare(b)
            assert safe_compare(a) >= safe_compare(b)
            assert safe_compare(a) > safe_compare(b)
        case (int() | float(), int() | float()) if a == 0 == b:
            neg_a, neg_b = repr(a) == '-0.0', repr(b) == '-0.0'
            assert (safe_compare(a) < safe_compare(b)) == (neg_a and not neg_b)
            assert (safe_compare(a) <= safe_compare(b)) == (neg_a or not neg_b)
            assert (safe_compare(a) > safe_compare(b)) == (neg_b and not neg_a)
            assert (safe_compare(a) >= safe_compare(b)) == (neg_b or not neg_a)
        case (int() | float(), int() | float()):
            assert (safe_compare(a) <= safe_compare(b)) == (a <= b)
            assert (safe_compare(a) < safe_compare(b)) == (a < b)
            assert (safe_compare(a) >= safe_compare(b)) == (a >= b)
            assert (safe_compare(a) > safe_compare(b)) == (a > b)


@pytest.mark.parametrize('a, b', itertools.product([-0.0, 0.0, 0], repeat=2))
def test_safe_compare_zeros(a: float, b: float) -> None:
    """Assert that `-0.0` is smaller than `0.0` and `0`."""
    neg_a, neg_b = repr(a) == '-0.0', repr(b) == '-0.0'
    lt = neg_a and not neg_b
    le = neg_a or not neg_b
    gt = neg_b and not neg_a
    ge = neg_b or not neg_a

    # Both sides
    assert (safe_compare(a) < safe_compare(b)) == lt
    assert (safe_compare(a) <= safe_compare(b)) == le
    assert (safe_compare(a) > safe_compare(b)) == gt
    assert (safe_compare(a) >= safe_compare(b)) == ge

    # Left side only
    assert (safe_compare(a) < b) == lt
    assert (safe_compare(a) <= b) == le
    assert (safe_compare(a) > b) == gt
    assert (safe_compare(a) >= b) == ge

    # Right side only
    assert (a < safe_compare(b)) == lt
    assert (a <= safe_compare(b)) == le
    assert (a > safe_compare(b)) == gt
    assert (a >= safe_compare(b)) == ge
