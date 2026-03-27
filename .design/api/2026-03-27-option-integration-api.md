# API Design: Option Type Integration into `contents()` and `arrays()`

- **Date:** 2026-03-27
- **Status:** Proposed
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

Option types are added as peer candidates alongside the existing wrapper types:

```python
_NodeType = Literal[
    'list', 'list_offset', 'record', 'regular', 'union',
    'indexed_option', 'byte_masked', 'bit_masked', 'unmasked',
]

candidates = list[_NodeType]()
# ...existing candidates...
if allow_indexed_option and allow_option_root:
    candidates.append('indexed_option')
if allow_byte_masked and allow_option_root:
    candidates.append('byte_masked')
if allow_bit_masked and allow_option_root:
    candidates.append('bit_masked')
if allow_unmasked and allow_option_root:
    candidates.append('unmasked')
```

### Recursion from Option Nodes

When an option type is selected, `contents()` recurses for the child content
with `allow_option_root=False` to prevent option-inside-option:

```python
case 'indexed_option':
    child = draw(recurse(max_size=max_size, allow_option_root=False))
    return draw(
        st_ak.contents.indexed_option_array_contents(
            child, max_length=max_length
        )
    )
```

The per-type `allow_*` flags are forwarded to `recurse` unchanged, so option
types can appear deeper in the tree (just not immediately inside another
option). `allow_option_root` is not included in the `recurse` partial — it
defaults to `True` in deeper recursive calls, except when explicitly overridden.

## `max_size`, `max_depth`, `max_length` Semantics

### `max_size` — Pass Through

Option types are transparent to the scalar budget. They pass `max_size` through
to their child content unchanged. No additional accounting is needed.

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

### The Constraint

`UnionArray` requires either all or none of its contents to be option types.
Mixed option/non-option contents raise `TypeError`.

### The Solution

When `contents()` generates a `UnionArray`, it flips a coin to decide whether
branches are option-wrapped:

- **No option**: Recurse for each branch with `allow_option_root=False`. Each
  branch is guaranteed non-option at its root. Options can still appear deeper.

- **All option**: Compose strategies so each branch is drawn as
  `option_contents(content=recurse_with_option_root_false)`. The
  `option_contents` strategy wraps each branch in a randomly chosen option type.
  The inner content is drawn by `recurse` with `allow_option_root=False`.

Both cases use strategies composed top-down. No post-processing is needed.

```python
case 'union':
    option_at_root = _any_option_allowed(...) and draw(st.booleans())
    if option_at_root:
        child_st = functools.partial(
            st_ak.contents.option_contents,
            content=functools.partial(
                recurse,
                allow_option_root=False,
            ),
            allow_indexed_option=allow_indexed_option,
            allow_byte_masked=allow_byte_masked,
            allow_bit_masked=allow_bit_masked,
            allow_unmasked=allow_unmasked,
        )
    else:
        child_st = functools.partial(recurse, allow_option_root=False)
    children = draw(
        content_lists(
            child_st,
            max_total_size=max_size,
            min_size=2,
        )
    )
    return draw(
        st_ak.contents.union_array_contents(children, max_length=max_length)
    )
```

The `_any_option_allowed()` helper checks whether at least one option type flag
is `True`. When all option types are disabled, the coin flip is skipped and
branches are always non-option.

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
