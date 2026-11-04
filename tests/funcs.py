"""Helpers shared by the strategy test modules."""

import inspect
from collections.abc import Callable, Mapping, Set
from typing import Any, Generic, NamedTuple, TypeVar

K = TypeVar('K')
D = TypeVar('D')


def assert_kwargs_match_signature(
    func: Callable[..., Any],
    *,
    exclude: Set[str] = frozenset(),
    kwargs_cls: type[Any],
    defaults: Mapping[str, Any],
    related_cls: type[Any] | None = None,
) -> None:
    """Raise unless the option declarations agree with `func`'s parameters.

    `kwargs_cls` and `related_cls` are the test module's `TypedDict` classes and
    `defaults` its `DEFAULTS`. Asserts that `kwargs_cls` has exactly the parameters
    of `func` other than `exclude`, that `defaults` equals the signature's default
    values, and that `related_cls`'s keys are a subset of `kwargs_cls`'s.

    Adapted from `tests/properties/operations/util.py` in awkward, with `exclude`
    in place of `data_param_names` and `related_cls` optional.
    """
    option_params = {
        name: p.default
        for name, p in inspect.signature(func).parameters.items()
        if name not in exclude
    }
    keys = kwargs_cls.__required_keys__ | kwargs_cls.__optional_keys__
    assert keys == set(option_params)

    assert defaults == option_params

    if related_cls is not None:
        related_keys = related_cls.__required_keys__ | related_cls.__optional_keys__
        assert related_keys <= keys


class KwargsDrawn(NamedTuple, Generic[K, D]):
    """Options for a strategy, with the values drawn in advance for them.

    `kwargs` are passed to the strategy under test. `drawn` holds, per option, the
    value drawn in the kwargs strategy that the result must contain, e.g. the content
    behind a `st.just(content)` option.
    """

    kwargs: K
    drawn: D
