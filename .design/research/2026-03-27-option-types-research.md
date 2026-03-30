# Option Types Research

- **Date:** 2026-03-27
- **Purpose:** Inform the design of option-type content strategies for
  hypothesis-awkward

## Overview

Awkward Array has four option-type Content classes that represent missing
(nullable) data. All four set `is_option = True` in their metadata. They differ
in how the mask is stored.

| Class                | Mask representation        | `is_indexed` |
| -------------------- | -------------------------- | ------------ |
| `IndexedOptionArray` | Index with negative = None | `True`       |
| `ByteMaskedArray`    | Byte mask (int8)           | `False`      |
| `BitMaskedArray`     | Bit-packed mask (uint8)    | `False`      |
| `UnmaskedArray`      | No mask (no actual nulls)  | `False`      |

Additionally, `IndexedArray` (`is_option = False`, `is_indexed = True`) is a
closely related non-option class that reorders/deduplicates content via an
index.

## Constructor Signatures

### IndexedOptionArray

```python
IndexedOptionArray(index, content, *, parameters=None)
```

- **`index`** — `Index32` or `Index64` (dtype `int32` or `int64`). Negative
  values mean "missing" (any negative, not just `-1`). Non-negative values index
  into content: `0 <= index[i] < len(content)`.
- **`content`** — A `Content` subtype.
- **`parameters`** — Optional dict.

Element semantics:

```python
if index[i] < 0:
    element = None
else:
    element = content[index[i]]
```

Length equals `len(index)`.

### ByteMaskedArray

```python
ByteMaskedArray(mask, content, valid_when, *, parameters=None)
```

- **`mask`** — `Index8` (dtype `int8`). One byte per element.
- **`content`** — A `Content` subtype.
- **`valid_when`** — `bool`. When `mask[i] == valid_when`, the element is valid;
  otherwise it is None.
- **`parameters`** — Optional dict.

Constraint: `len(mask) <= len(content)`. Length equals `len(mask)`.

### BitMaskedArray

```python
BitMaskedArray(mask, content, valid_when, length, lsb_order, *, parameters=None)
```

- **`mask`** — `IndexU8` (dtype `uint8`). Bits packed 8 per byte.
- **`content`** — A `Content` subtype.
- **`valid_when`** — `bool`. A bit equal to `valid_when` means valid.
- **`length`** — Non-negative `int`. The logical array length.
- **`lsb_order`** — `bool`. `True` = least-significant bit first (Arrow
  convention); `False` = most-significant bit first.
- **`parameters`** — Optional dict.

Constraint: `length <= len(mask) * 8` and `length <= len(content)`. Length
equals `length`.

Bit extraction:

```python
# lsb_order=True
is_valid[j] = bool(mask[j // 8] & (1 << (j % 8))) == valid_when

# lsb_order=False
is_valid[j] = bool(mask[j // 8] & (128 >> (j % 8))) == valid_when
```

### UnmaskedArray

```python
UnmaskedArray(content, *, parameters=None)
```

- **`content`** — A `Content` subtype.
- **`parameters`** — Optional dict.

No mask buffer. All values are present. Produces `OptionType` in the type system
but never has actual nulls. Length equals `len(content)`.

### IndexedArray (non-option)

```python
IndexedArray(index, content, *, parameters=None)
```

- **`index`** — `Index32`, `IndexU32`, or `Index64` (dtype `int32`, `uint32`, or
  `int64`). All values must be non-negative.
- **`content`** — A `Content` subtype.
- **`parameters`** — Optional dict.

Same structure as `IndexedOptionArray` but negative indices are not allowed and
the array has no option semantics. Used for reordering, deduplication, and
categorical data (`parameters={"__array__": "categorical"}`).

## Validation Rules (All Option Types)

All four option-type constructors enforce the same content restriction:

```python
if content.is_union or content.is_indexed or content.is_option:
    raise TypeError(
        "... cannot contain a union-type, option-type, or indexed 'content' ...; "
        "try ... .simplified instead"
    )
```

**Cannot directly wrap:**

- Another option type (`ByteMaskedArray`, `BitMaskedArray`, `UnmaskedArray`,
  `IndexedOptionArray`)
- An indexed type (`IndexedArray`, `IndexedOptionArray`)
- A union type (`UnionArray`)

**Exception:** `IndexedOptionArray` and `IndexedArray` allow wrapping
`UnionArray` when `parameters={"__array__": "categorical"}`.

**Can directly wrap:** `NumpyArray`, `EmptyArray`, `RegularArray`,
`ListOffsetArray`, `ListArray`, `RecordArray`.

Each class provides a `simplified()` classmethod that handles invalid nesting by
flattening to `IndexedOptionArray`.

### Type-Specific Validation

| Check                        | IndexedOptionArray | ByteMaskedArray | BitMaskedArray     | UnmaskedArray |
| ---------------------------- | ------------------ | --------------- | ------------------ | ------------- |
| Index/mask dtype             | int32, int64       | int8            | uint8              | (none)        |
| `valid_when` must be bool    | —                  | Yes             | Yes                | —             |
| `lsb_order` must be bool     | —                  | —               | Yes                | —             |
| `length` must be non-neg int | —                  | —               | Yes                | —             |
| `len(mask) <= len(content)`  | —                  | Yes             | `length <= len(c)` | —             |
| `length <= len(mask) * 8`    | —                  | —               | Yes                | —             |

## Nesting Rules

### Option inside other content types

| Outer type        | Option content allowed? |
| ----------------- | ----------------------- |
| `RegularArray`    | Yes                     |
| `ListOffsetArray` | Yes                     |
| `ListArray`       | Yes                     |
| `RecordArray`     | Yes (any field)         |
| `UnionArray`      | Yes, but ALL or NONE    |
| Another option    | No                      |
| `IndexedArray`    | No                      |

### UnionArray "all or none" rule

`UnionArray` requires that either all contents are option types or none are.
Mixed option/non-option contents raise `TypeError`. When using
`UnionArray.simplified()`, non-option contents are automatically wrapped in
`UnmaskedArray` to satisfy this rule.

### Option cannot nest inside option

No option type can directly wrap another option type. This means a strategy for
option content should wrap non-option content and avoid double-wrapping.

## Scalar Counting and Size Budget

Option types are **transparent to leaf scalar counting** (`max_leaf_size`): they
do not add leaf scalars themselves. The leaf scalars come from the wrapped
content. An `IndexedOptionArray` wrapping a `NumpyArray` of length 5 has 5 leaf
scalars regardless of how many entries are masked as None.

However, option types with buffers (index, mask) are **not** transparent to
total `content_size()` (`max_size`). See
[option-integration-api](../api/2026-03-27-option-integration-api.md#max_size--deduct-buffer-overhead)
for `content_size()` formulas.

For `IndexedOptionArray`, the index can reference content elements multiple
times or skip some entirely. The leaf scalar count comes from the content, not
from `len(index)`.

For `ByteMaskedArray` and `BitMaskedArray`, the content may be longer than the
mask. Extra content elements beyond `len(mask)` (or `length` for
`BitMaskedArray`) are not accessible through the array but still exist in
memory.

## Generation Considerations

### IndexedOptionArray

The index array needs:

1. A length (number of elements in the option array).
2. For each position, either a negative value (missing) or a valid index into
   content (`0 <= i < len(content)`).

A simple strategy: draw a content, then draw an index array where each entry is
either `-1` or a random valid index. The fraction of missing values could be
controlled or left to Hypothesis.

Index can reference content elements out of order and can reference the same
element multiple times (deduplication). For simplicity, compact indexing
(`index[i] = i` for valid entries) is sufficient and avoids needing the content
to be larger than the array.

### ByteMaskedArray

Draw a content, then draw a boolean mask of the same length. Convert booleans to
int8. Choose `valid_when` from `{True, False}`.

### BitMaskedArray

Draw a content, then draw a uint8 mask with `ceil(length / 8)` bytes. Choose
`valid_when` and `lsb_order` from `{True, False}`.

### UnmaskedArray

Trivial: just wrap a content. No additional buffers needed.

### Where in the tree

Option nodes should appear between the "wrapper" nodes (list, regular, record,
union) and their children. An option node wraps non-option content. The tree
structure looks like:

```text
ListOffsetArray
  └── IndexedOptionArray   ← option wraps a non-option
        └── NumpyArray
```

Not:

```text
IndexedOptionArray         ← option at root is fine
  └── ListOffsetArray
        └── NumpyArray
```

Both positions are valid in Awkward Array. The strategy should allow option
nodes at any position in the tree, subject to the "no option inside option"
constraint.

### Integration with UnionArray

If option types are enabled, `UnionArray` contents might be option-typed. Per
the "all or none" rule, the strategy must ensure either all union branches are
wrapped in option types or none are. The simplest approach: when generating
union contents, decide once whether to use option-wrapped branches, then apply
consistently.
