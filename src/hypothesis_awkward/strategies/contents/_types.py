from typing import Protocol

from hypothesis import strategies as st

from awkward.contents import Content


class StContent(Protocol):
    """Callable that returns a strategy for a single content within a size budget.

    The `*_from_contents()` strategies call it to draw their children. `contents()`
    passes itself with its options bound by `functools.partial` and `max_depth`
    reduced by one, so that the children follow the caller's options.

    Other callables can be passed, e.g., one that draws only leaves. A callable that
    never produces the roots the `allow_*_root` flags exclude can accept them with
    `**kwargs` and ignore them.
    """

    def __call__(
        self,
        *,
        max_size: int,
        max_leaf_size: int | None,
        min_length: int = ...,
        max_length: int | None = ...,
        allow_union_root: bool = ...,
        allow_option_root: bool = ...,
        allow_indexed_root: bool = ...,
        allow_categorical_root: bool = ...,
    ) -> st.SearchStrategy[Content]: ...


class StOption(Protocol):
    """Callable that wraps content drawn from a `StContent` in an option type.

    `union_array_from_contents()` uses it, through `content_lists()`, to make all
    children of a union option types when the first child is one, since Awkward
    requires either all or none of a union's children to be option types.

    `contents()` passes `option_from_contents()` with its `allow_*` option flags bound
    by `functools.partial`, or `None` when the flags exclude every option type.
    """

    def __call__(
        self,
        content: StContent,
        *,
        max_size: int,
        max_leaf_size: int | None,
    ) -> st.SearchStrategy[Content]: ...
