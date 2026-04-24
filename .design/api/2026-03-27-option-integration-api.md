# API Design: Option Type Integration into `contents()` and `arrays()`

- **Date:** 2026-03-27
- **Status:** Implemented — all four `allow_*` flags, `allow_option_root`, the
  `*_from_contents` bridges, option-aware `UnionArray` coordination, and the
  test/util updates described below are present in the current code.
- **Author:** Claude (with developer collaboration)

## Overview

This document describes how option types are integrated into the `contents()`
tree builder and the `arrays()` public API. It covers the new flags, depth/size
semantics, and the UnionArray coordination.

See [option-contents-api](2026-03-27-option-contents-api.md) for the per-type
option strategies. See
[option-types-research](../research/2026-03-27-option-types-research.md) for
background.

## Changes to `contents()`

### New Parameters

Four per-type flags, following the existing `allow_*` pattern:

```python
@st.composite
def contents(
    draw: st.DrawFn,
    *,
    # ...existing parameters...
    allow_indexed_option: bool = True,
    allow_byte_masked: bool = True,
    allow_bit_masked: bool = True,
    allow_unmasked: bool = True,
    allow_option_root: bool = True,
    # ...existing parameters...
) -> Content:
```

- **`allow_indexed_option`** — No `IndexedOptionArray` anywhere in the tree if
  `False`.
- **`allow_byte_masked`** — No `ByteMaskedArray` anywhere in the tree if
  `False`.
- **`allow_bit_masked`** — No `BitMaskedArray` anywhere in the tree if `False`.
- **`allow_unmasked`** — No `UnmaskedArray` anywhere in the tree if `False`.
- **`allow_option_root`** — The outermost content node cannot be an option type
  if `False`. Does not affect deeper levels. Analogous to `allow_union_root`.

### Node Type Selection

Option types are added as peer candidates alongside the existing wrapper types.
Each option type gets a `*_from_contents` bridge function that conforms to the
`_StFromContents` protocol, matching the pattern used by
`list_offset_array_from_contents`, `record_array_from_contents`, etc.:

```python
candidates = list[_StFromContents]()
# ...existing candidates...
if allow_indexed_option and allow_option_root:
    candidates.append(indexed_option_array_from_contents)
if allow_byte_masked and allow_option_root:
    candidates.append(byte_masked_array_from_contents)
if allow_bit_masked and allow_option_root:
    candidates.append(bit_masked_array_from_contents)
if allow_unmasked and allow_option_root:
    candidates.append(unmasked_array_from_contents)
```

### Recursion from Option Nodes

Each `*_from_contents` function receives a `StContent` callable (the partially
applied `recurse`) and calls it with `allow_option_root=False` to prevent
option-inside-option. For example:

```python
@st.composite
def indexed_option_array_from_contents(
    draw: st.DrawFn,
    content: StContent,
    *,
    max_size: int,
    max_leaf_size: int | None,
    max_length: int | None,
) -> IndexedOptionArray:
    ml = max_length if max_length is not None else max_size
    n = draw(st.integers(min_value=0, max_value=ml))
    max_content_size = max(max_size - n, 0)  # deduct index overhead
    child = draw(content(max_size=max_content_size, max_leaf_size=max_leaf_size, allow_option_root=False))
    result = draw(indexed_option_array_contents(child, max_length=n))
    assume(content_size(result) <= max_size)
    return result
```

The per-type `allow_*` flags are forwarded to `recurse` unchanged, so option
types can appear deeper in the tree (just not immediately inside another
option). `allow_option_root` is not included in the `recurse` partial — it
defaults to `True` in deeper recursive calls, except when explicitly overridden.

## `max_size`, `max_leaf_size`, `max_depth`, `max_length` Semantics

### `max_size` — Deduct Buffer Overhead

Option types with buffers (index, mask) are **not** transparent to `max_size`.
Each `_from_contents` function must deduct its buffer overhead before passing
`max_size` to the child, following the same pattern as
`list_offset_array_from_contents`.

`content_size()` formulas for option types:

```text
content_size(IndexedOptionArray)  = len(index) + content_size(child)
content_size(ByteMaskedArray)     = len(mask) + content_size(child)
content_size(BitMaskedArray)      = ceil(length / 8) + content_size(child)
content_size(UnmaskedArray)       = content_size(child)
```

Only `UnmaskedArray` passes `max_size` through unchanged.

### `max_leaf_size` — Pass Through

Option types are transparent to the leaf scalar budget. They pass
`max_leaf_size` through to their child content unchanged. No additional
accounting is needed.

### `max_depth` — Not Consumed

Option nodes do not consume a depth level. An option wrapper is lightweight (no
extra scalars, trivial traversal). This matches how string/bytestring nodes do
not consume depth.

In the worst case, each non-option node at every depth level could be wrapped in
one option node (since option-inside-option is forbidden). A `max_depth=5` tree
could have up to 10 actual nodes deep (5 non-option + 5 option wrappers).

### `max_length` — Root Only

`max_length` is applied to the outermost node only, consistent with existing
behavior. A root-level option node respects `max_length` by constraining its own
length (index length, mask length, or content length depending on the type).
Deeper option nodes have no explicit `max_length` constraint — their length is
determined by their parent.

## UnionArray Coordination

Per the "all or none" rule (see
[option-types-research](../research/2026-03-27-option-types-research.md#unionarray-all-or-none-rule)),
`UnionArray` requires either all or none of its contents to be option types.

When `contents()` generates a `UnionArray`, it flips a coin to decide whether
branches are option-wrapped:

- **No option**: Recurse for each branch with `allow_option_root=False`. Each
  branch is guaranteed non-option at its root. Options can still appear deeper.

- **All option**: Compose strategies so each branch is drawn as
  `option_contents(content=recurse_with_option_root_false)`. The
  `option_contents` strategy wraps each branch in a randomly chosen option type.
  The inner content is drawn by `recurse` with `allow_option_root=False`.

Both cases use strategies composed top-down. No post-processing is needed.

`contents()` handles the coordination by partially applying the option flags to
`union_array_from_contents` before adding it to the candidates list:

```python
if allow_union and allow_union_root:
    candidates.append(functools.partial(
        union_array_from_contents,
        allow_indexed_option=allow_indexed_option,
        allow_byte_masked=allow_byte_masked,
        allow_bit_masked=allow_bit_masked,
        allow_unmasked=allow_unmasked,
    ))
```

### Why Not Post-Process

Post-processing (draw children, then fix mixed results) generates content before
knowing whether it's valid. The top-down approach composes strategies so valid
content is generated by construction. This is consistent with how the rest of
`contents()` works.

## Changes to `arrays()`

Forward the new flags:

```python
@st.composite
def arrays(
    draw: st.DrawFn,
    *,
    # ...existing parameters...
    allow_indexed_option: bool = True,
    allow_byte_masked: bool = True,
    allow_bit_masked: bool = True,
    allow_unmasked: bool = True,
    # ...existing parameters...
) -> ak.Array:
```

`allow_option_root` is not exposed on `arrays()` — it is an internal mechanism
for preventing option-inside-option, not a user-facing control.

## Changes to `_nesting_depth()` in Tests

Option nodes should not add to the nesting depth:

```python
if isinstance(c, (IndexedOptionArray, ByteMaskedArray, BitMaskedArray, UnmaskedArray)):
    return _nesting_depth(c.content)
```

## Changes to Utility Functions

`iter_contents()` in `util/awkward.py` already handles `IndexedOptionArray` and
`UnmaskedArray`. Add `ByteMaskedArray` and `BitMaskedArray`:

```python
case (
    BitMaskedArray()
    | ByteMaskedArray()
    | IndexedOptionArray()
    | ListArray()
    | ListOffsetArray()
    | RegularArray()
    | UnmaskedArray()
):
    yield item
    stack.append(item.content)
```

## Design Decisions

### 1. Per-Type Flags, Not a Single `allow_option`

**Decision:** Four separate `allow_*` flags instead of a single `allow_option`.

**Rationale:** Consistent with how all other content types have their own flag.
Users testing specific code paths (e.g., ByteMaskedArray handling) can enable
only the relevant type. A combined `allow_option` convenience flag can be added
later if needed.

### 2. `allow_option_root` Analogous to `allow_union_root`

**Decision:** `allow_option_root` controls whether the outermost node can be an
option type. It does not propagate to deeper levels.

**Rationale:** Prevents option-inside-option at construction time. When
recursing from an option node, `allow_option_root=False` ensures the child is
non-option. Deeper recursive calls default to `allow_option_root=True`, so
options can appear at any depth subject to the no-double-wrapping rule.

### 3. Option Nodes Do Not Consume Depth

**Decision:** Option wrappers do not decrement `max_depth`.

**Rationale:** Option wrappers are lightweight — no extra scalars, trivial
traversal. Consuming depth would reduce the variety of generated structures. At
worst, each depth level gets one option wrapper, roughly doubling the node count
for a given `max_depth`.

### 4. UnionArray Coordination via Strategy Composition

**Decision:** When generating `UnionArray` children, flip a coin once to decide
all-option or no-option. Compose strategies top-down.

**Rationale:** Generates valid content by construction. The "all option" case
composes `option_contents(content=recurse)`, and the "no option" case uses
`recurse(allow_option_root=False)`. No mixed results to fix.

### 5. `allow_option_root` Not Exposed on `arrays()`

**Decision:** `allow_option_root` is internal to `contents()`.

**Rationale:** It exists to prevent option-inside-option nesting. Users of
`arrays()` control option generation via the four per-type flags. The root
constraint is an implementation detail of the tree builder.
