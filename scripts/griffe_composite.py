"""Griffe extension that documents `@st.composite` functions by their call signature.

`hypothesis.strategies.composite` turns a function into one that takes the same
arguments except the first (`draw`) and returns a strategy for the original return
type. Griffe reads the source, so it sees the function as written. This extension
applies the same change, so that the API reference shows how the function is called.

Enabled in `zensical.toml`, in the `extensions` option of the mkdocstrings Python
handler.
"""

import ast
from typing import Any

import griffe

_COMPOSITE = frozenset({'st.composite', 'composite', 'hypothesis.strategies.composite'})


class Composite(griffe.Extension):
    """Remove the `draw` parameter and return `st.SearchStrategy[<return type>]`."""

    def on_function_instance(self, *, func: griffe.Function, **kwargs: Any) -> None:
        if not any(str(d.value) in _COMPOSITE for d in func.decorators):
            return
        names = [p.name for p in func.parameters]
        if names and names[0] == 'draw':
            del func.parameters['draw']
        if func.returns is None or func.parent is None:
            return
        node = ast.parse('st.SearchStrategy[_]', mode='eval').body
        returns = griffe.safe_get_annotation(node, parent=func.parent)
        if isinstance(returns, griffe.ExprSubscript):
            returns.slice = func.returns
            func.returns = returns
