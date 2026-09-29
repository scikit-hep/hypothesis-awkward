from hypothesis import given
from hypothesis import strategies as st

from hypothesis_awkward.util import safe_compare


def test_repr() -> None:
    assert repr(safe_compare(1)) == '1'
    assert repr(safe_compare(None)) == 'GreaterAndLessThanAny()'


@given(st.data())
def test_safe_compare(data: st.DataObject) -> None:
    a = data.draw(st.none() | st.integers())
    b = data.draw(st.none() | st.integers() | st.just(a))

    match a, b:
        case (None, _) | (_, None):
            assert safe_compare(a) <= safe_compare(b)
            assert safe_compare(a) < safe_compare(b)
            assert safe_compare(a) >= safe_compare(b)
            assert safe_compare(a) > safe_compare(b)
        case int(), int():
            assert (safe_compare(a) <= safe_compare(b)) == (a <= b)
            assert (safe_compare(a) < safe_compare(b)) == (a < b)
            assert (safe_compare(a) >= safe_compare(b)) == (a >= b)
            assert (safe_compare(a) > safe_compare(b)) == (a > b)
