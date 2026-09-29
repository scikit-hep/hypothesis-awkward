from hypothesis import given
from hypothesis import strategies as st

from hypothesis_awkward.util import safe_max, safe_min
from hypothesis_awkward.util.safe import _signed_max, _signed_min

st_integers = st.integers()
st_floats = st.floats(allow_nan=False)


@given(st.data())
def test_safe_min(data: st.DataObject) -> None:
    st_ = data.draw(st.sampled_from([st_integers, st_floats]))
    num_vals = data.draw(st.lists(st_))
    none_vals = data.draw(st.lists(st.none()))
    vals = data.draw(st.permutations(num_vals + none_vals))  # type: ignore

    default_val = data.draw(st.one_of(st_, st.none()))

    result = safe_min(vals, default=default_val)

    expected = _signed_min(num_vals, default=default_val)

    assert repr(result) == repr(expected)


@given(st.data())
def test_safe_max(data: st.DataObject) -> None:
    st_ = data.draw(st.sampled_from([st_integers, st_floats]))
    num_vals = data.draw(st.lists(st_))
    none_vals = data.draw(st.lists(st.none()))
    vals = data.draw(st.permutations(num_vals + none_vals))  # type: ignore

    default_val = data.draw(st.one_of(st_, st.none()))

    result = safe_max(vals, default=default_val)

    expected = _signed_max(num_vals, default=default_val)

    assert repr(result) == repr(expected)
