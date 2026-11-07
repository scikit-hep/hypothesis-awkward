from typing import Protocol

from hypothesis import strategies as st

from awkward.contents import Content


class StContent(Protocol):
    """Callable that returns a strategy for a single content within a size budget."""

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
    """Callable that wraps content drawn from a `StContent` in an option type."""

    def __call__(
        self,
        content: StContent,
        *,
        max_size: int,
        max_leaf_size: int | None,
    ) -> st.SearchStrategy[Content]: ...
