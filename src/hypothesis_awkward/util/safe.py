import math
from collections.abc import Iterable
from typing import Any, Optional, TypeVar

T = TypeVar('T')


def safe_min(vals: Iterable[T], default: Optional[T] = None) -> Optional[T]:
    """The smallest item in `vals` that is not `None`.

    Parameters
    ----------
    vals
        An iterable of values.
    default
        The value to return if `vals` is empty or all items are `None`.

    Examples
    --------
    >>> safe_min([None, 1, 2, None])
    1

    It returns `None` if `vals` is empty or all items in `vals` are `None`.

    >>> print(safe_min([None, None]))
    None

    >>> print(safe_min([]))
    None

    If `default` is given, it returns `default` instead of `None`.

    >>> safe_min([None, None], default=-1)
    -1

    >>> safe_min([], default=-1)
    -1

    `-0.0` is smaller than `0.0`, as in the bounds of `st.floats()`.

    >>> safe_min([0.0, None, -0.0])
    -0.0
    """
    return _signed_min((v for v in vals if v is not None), default=default)


def safe_max(vals: Iterable[T], default: Optional[T] = None) -> Optional[T]:
    """The largest item in `vals` that is not `None`.

    Parameters
    ----------
    vals
        An iterable of values.
    default
        The value to return if `vals` is empty or all items are `None`.

    Examples
    --------
    >>> safe_max([None, 1, 2, None])
    2

    It returns `None` if `vals` is empty or all items in `vals` are `None`.

    >>> print(safe_max([None, None]))
    None

    >>> print(safe_max([]))
    None

    If `default` is given, it returns `default` instead of `None`.

    >>> safe_max([None, None], default=-1)
    -1

    >>> safe_max([], default=-1)
    -1

    `0.0` is larger than `-0.0`, as in the bounds of `st.floats()`.

    >>> safe_max([-0.0, None, 0.0])
    0.0
    """
    return _signed_max((v for v in vals if v is not None), default=default)


class GreaterAndLessThanAny:
    """True for all inequality comparisons.

    Examples
    --------
    >>> GreaterAndLessThanAny() < 1
    True

    >>> GreaterAndLessThanAny() > 1
    True

    >>> GreaterAndLessThanAny() <= 1
    True

    >>> GreaterAndLessThanAny() >= 1
    True
    """

    def __le__(self, _: Any) -> bool:
        return True

    def __lt__(self, _: Any) -> bool:
        return True

    def __ge__(self, _: Any) -> bool:
        return True

    def __gt__(self, _: Any) -> bool:
        return True

    # def __eq__(self, _: Any) -> bool:
    #     return True

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


def safe_compare(value: T | None) -> T | GreaterAndLessThanAny:
    """Return `value` if not `None`, else an object true for all comparisons.

    This function helps you concisely write assertions that compare
    values that may be `None`.

    Parameters
    ----------
    value
        A value or `None`.

    Examples
    --------
    Suppose you have `min_` and `max_` that may be `None`

    >>> import random
    >>> min_ = random.choice([None, 1])
    >>> max_ = random.choice([None, 3])

    and `val` that should be in the range `[min_, max_]`:

    >>> val = 2

    Without this function, you need to check if `min_` and `max_`
    are `None`.

    >>> if min_ is not None:
    ...     assert min_ <= val

    >>> if max_ is not None:
    ...     assert val <= max_

    This function lets you write the same assertion in one line:

    >>> assert safe_compare(min_) <= val <= safe_compare(max_)
    """
    if value is None:
        return GreaterAndLessThanAny()
    return value


def _signed_min(vals: Iterable[T], default: Optional[T] = None) -> Optional[T]:
    """Like `min()`, but orders `-0.0` before `0.0`.

    `min()` treats `-0.0` and `0.0` as equal and returns whichever comes first.
    Hypothesis's float bounds treat `-0.0` as smaller, e.g., `st.floats(min_value=0.0)`
    never generates `-0.0`.

    Examples
    --------
    >>> _signed_min([0.0, -0.0])
    -0.0

    >>> print(_signed_min([]))
    None
    """
    return min(vals, key=_sign_aware_key, default=default)


def _signed_max(vals: Iterable[T], default: Optional[T] = None) -> Optional[T]:
    """Like `max()`, but orders `-0.0` before `0.0`.

    Examples
    --------
    >>> _signed_max([0.0, -0.0])
    0.0

    >>> _signed_max([-0.0, 0.0])
    0.0

    >>> print(_signed_max([]))
    None
    """
    return max(vals, key=_sign_aware_key, default=default)


def _sign_aware_key(v: Any) -> tuple[Any, float]:
    """Sort key that breaks ties between zeros by their signs."""
    return (v, math.copysign(1.0, v) if v == 0 else 0.0)
