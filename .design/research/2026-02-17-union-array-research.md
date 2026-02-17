# UnionArray Research

Date: 2026-02-17

## Overview

`UnionArray` is a multi-content node in Awkward Array that represents a tagged
union (sum type). Each element selects one of several alternative content arrays
via a `(tags, index)` pair. Unlike `RecordArray` (which combines all fields into
every element), `UnionArray` picks exactly one alternative per element.

## Constructor Signature

```python
UnionArray(tags, index, contents, *, parameters=None)
```

Parameters:

- **`tags`** — An `Index8` (dtype=int8) array. Each value selects which content
  the element comes from: `0 <= tags[i] < len(contents)`.
- **`index`** — An `Index32`, `IndexU32`, or `Index64` array. Each value selects
  which element within the chosen content: `0 <= index[i] < len(contents[tags[i]])`.
- **`contents`** — An iterable of `Content` subclasses. Minimum 2 contents
  required (raises `TypeError` otherwise).
- **`parameters`** — Optional dict. No special keys (unlike `RecordArray`'s
  `__record__`).

The element at position `i` is `contents[tags[i]][index[i]]`.

## Validation Rules

The constructor enforces:

1. `tags` must have dtype=int8
2. `index` must have dtype in {int32, uint32, int64}
3. `len(contents) >= 2` (raises `TypeError`)
4. No union-type content (no nested unions)
5. No non-categorical indexed-type content
6. Option-type contents: either ALL or NONE must be option (raises `TypeError` if
   mixed)
7. `len(tags) <= len(index)` (usually equal; extra index values are unreachable)

## Construction Examples

### Two numeric types

```python
u = ak.contents.UnionArray(
    ak.index.Index8(np.array([0, 1, 0, 1, 0], dtype=np.int8)),
    ak.index.Index64(np.array([0, 0, 1, 1, 2])),
    [
        ak.contents.NumpyArray(np.array([1.1, 2.2, 3.3])),
        ak.contents.NumpyArray(np.array([10, 20], dtype=np.int64)),
    ],
)
# [1.1, 10, 2.2, 20, 3.3]
# Type: 5 * union[float64, int64]
```

### Mixed scalar and list

```python
u = ak.contents.UnionArray(
    ak.index.Index8(np.array([0, 1, 0, 1], dtype=np.int8)),
    ak.index.Index64(np.array([0, 0, 1, 1])),
    [
        ak.contents.NumpyArray(np.array([1.0, 2.0])),
        ak.from_iter([[10, 20], [30]], highlevel=False),
    ],
)
# [1.0, [10, 20], 2.0, [30]]
# Type: 4 * union[float64, var * int64]
```

### With record contents

```python
u = ak.contents.UnionArray(
    ak.index.Index8(np.array([0, 1, 0, 1], dtype=np.int8)),
    ak.index.Index64(np.array([0, 0, 1, 1])),
    [
        ak.contents.RecordArray(
            [ak.contents.NumpyArray(np.array([1.0, 2.0]))],
            fields=['x'],
        ),
        ak.contents.RecordArray(
            [ak.contents.NumpyArray(np.array([10, 20], dtype=np.int64))],
            fields=['y'],
        ),
    ],
)
# [{x: 1.0}, {y: 10}, {x: 2.0}, {y: 20}]
# Type: 4 * union[{x: float64}, {y: int64}]
```

### Zero-length union

```python
u = ak.contents.UnionArray(
    ak.index.Index8(np.array([], dtype=np.int8)),
    ak.index.Index64(np.array([], dtype=np.int64)),
    [
        ak.contents.NumpyArray(np.array([], dtype=np.float64)),
        ak.contents.NumpyArray(np.array([], dtype=np.int64)),
    ],
)
# Type: 0 * union[float64, int64]
```

### Via `ak.from_iter` (heterogeneous Python data)

```python
u = ak.from_iter([1.1, [1, 2], 'hello', 3.3])
# Type: 4 * union[float64, var * int64, string]
```

### Via `ak.concatenate` with incompatible types

```python
c = ak.concatenate([ak.Array([1.0, 2.0]), ak.Array([[1, 2], [3]])])
# Type: 4 * union[float64, var * int64]
```

## Key Properties

| Property   | Returns                      | Notes                          |
| ---------- | ---------------------------- | ------------------------------ |
| `tags`     | `Index8` object              | `len(tags)` = array length     |
| `index`    | `Index` (int32/uint32/int64) | `len(index) >= len(tags)`      |
| `contents` | `list[Content]`              | The alternative content arrays |
| `length`   | `ShapeItem`                  | Equal to `len(tags)`           |
| `is_union` | `True`                       | Identity check                 |

Key methods:

| Method         | Description                                                 |
| -------------- | ----------------------------------------------------------- |
| `content(i)`   | Access content at index `i`                                 |
| `project(i)`   | Extract only elements for tag `i` (filtering operation)     |
| `simplified()` | Class method: flatten nested unions, merge compatible types |

## Nesting Rules

`UnionArray` has **asymmetric** nesting restrictions — strict rules on what can
contain it and what it can contain.

### Where UnionArray can appear (as child)

| Parent context       | Allowed? | Notes                             |
| -------------------- | -------- | --------------------------------- |
| `ListOffsetArray`    | Yes      |                                   |
| `ListArray`          | Yes      |                                   |
| `RegularArray`       | Yes      |                                   |
| `RecordArray`        | Yes      |                                   |
| `IndexedArray`       | **No**   | Cannot contain union-type content |
| `IndexedOptionArray` | **No**   | Cannot contain union-type content |
| `ByteMaskedArray`    | **No**   | Cannot contain union-type content |
| `BitMaskedArray`     | **No**   | Cannot contain union-type content |
| `UnmaskedArray`      | **No**   | Cannot contain union-type content |
| `UnionArray`         | **No**   | No nested unions                  |

Key takeaway: UnionArray can only appear inside list nodes and RecordArray. It
**cannot** appear inside option nodes, indexed nodes, or other unions.

### What UnionArray can contain (as parent)

| Child content type   | Allowed?    | Notes                                          |
| -------------------- | ----------- | ---------------------------------------------- |
| `NumpyArray`         | Yes         |                                                |
| `EmptyArray`         | Yes         | Produces `union[..., unknown]`                 |
| `ListOffsetArray`    | Yes         | Including string/bytestring                    |
| `ListArray`          | Yes         |                                                |
| `RegularArray`       | Yes         |                                                |
| `RecordArray`        | Yes         |                                                |
| `UnionArray`         | **No**      | No nested unions                               |
| `IndexedArray`       | **No**      | Unless categorical (`__array__="categorical"`) |
| Option nodes (all 4) | Conditional | Only if ALL contents are option-type           |

The "all or none" option rule means: either every content in the union is an
option node, or none of them is. Mixed option/non-option contents raise
`TypeError`.

**Indirect nesting is valid:** `Union → Record → Union` and `Union → List →
Union` are both fine because the "no nested union" rule applies only to direct
children.

## Tags and Index Mechanics

### Tags

- **Dtype:** int8 only (via `ak.index.Index8`)
- **Valid range:** `0 <= tags[i] < len(contents)` for all `i`
- **Maximum contents:** 128 (int8 supports 0–127)
- **Length semantics:** `len(tags)` = `len(union)`

### Index

- **Dtype options:** int32 (via `Index32`), uint32 (via `IndexU32`), or int64
  (via `Index64`)
- **Valid range:** `0 <= index[i] < len(contents[tags[i]])` for all `i`
- **Constraint:** `len(index) >= len(tags)` (usually equal)

### Compact vs. sparse indexing

- **Compact:** Each content element is referenced exactly once. Index values form
  a dense range `[0, 1, ..., count_k-1]` per tag `k`.
- **Sparse:** Contents can have unreferenced elements. Index values may skip
  entries.

Both are valid. Compact indexing is simpler for strategy generation.

## Length Coordination

- `len(UnionArray)` = `len(tags)`
- `len(index) >= len(tags)` (required)
- Each content can be **any** length >= the maximum index value referencing it + 1
- Contents do **not** need to share the same length

### Key difference from RecordArray

| Aspect              | RecordArray                        | UnionArray                          |
| ------------------- | ---------------------------------- | ----------------------------------- |
| Length coordination | All contents share record length   | Each content independently sized    |
| Element access      | All contents contribute to element | One content contributes per element |
| Minimum contents    | 0                                  | 2                                   |

## UnionArray vs. RecordArray

Both are multi-content nodes but represent fundamentally different algebraic
types:

| Aspect                  | RecordArray (product type)   | UnionArray (sum type)                         |
| ----------------------- | ---------------------------- | --------------------------------------------- |
| **Semantics**           | Every element has ALL fields | Every element is ONE of the types             |
| **Algebraic type**      | Product (AND)                | Sum (OR)                                      |
| **Min contents**        | 0                            | 2                                             |
| **Has field names?**    | Yes (named) or None (tuple)  | No                                            |
| **Has buffers?**        | No                           | Yes: `tags` (int8) + `index`                  |
| **Length coordination** | All contents share length    | Each content independent                      |
| **Nesting as child**    | No restrictions              | Cannot be inside option/indexed/union         |
| **Nesting as parent**   | No restrictions              | No union/indexed children; all-or-none option |
| **Form class**          | `RecordForm`                 | `UnionForm`                                   |
| **Type class**          | `RecordType`                 | `UnionType`                                   |

## Integration with the Tree Builder

The bottom-up tree builder in `contents()` was designed with UnionArray in mind.
The existing algorithm already supports multi-child nodes via the "another edge?"
mechanism (currently used only by RecordArray).

### What needs to change

1. **`can_branch`**: Currently `allow_record`. Becomes
   `allow_record or allow_union`.

2. **Multi-child node type selection**: Currently, 2+ children always means
   RecordArray. With UnionArray, the node type must be drawn from the enabled
   multi-child types:

   | Children | Possible node types (with UnionArray)                 |
   | -------- | ----------------------------------------------------- |
   | 1        | RegularArray, ListOffsetArray, ListArray, RecordArray |
   | 2+       | RecordArray, UnionArray                               |

3. **Minimum children enforcement**: RecordArray works with 1+ children, but
   UnionArray requires 2+. When UnionArray is selected, at least 2 children must
   exist. Options:
   - Draw node type first, then enforce minimum children
   - Draw children first, then constrain node type (current approach — easier)

4. **Buffer generation**: RecordArray has no buffers. UnionArray needs `(tags,
   index)` arrays.

### Buffer generation strategy

The simplest approach for valid `(tags, index)` pairs:

1. Given `n` children with lengths `L_0, L_1, ..., L_{n-1}`:
2. Total union length = `sum(L_k)` (compact indexing — every element referenced)
3. For each content `k`, create `L_k` entries: `tags = [k] * L_k`,
   `index = [0, 1, ..., L_k - 1]`
4. Concatenate all entries
5. Shuffle (random permutation) to interleave

This guarantees:

- Every content element is referenced exactly once
- All tags are valid (0 to n-1)
- All index values are valid (0 to L_k - 1)
- The union exercises all contents

### Length considerations

Unlike RecordArray (where all contents must share the record length), UnionArray
contents can have different lengths. The union length is determined by `len(tags)`,
which with compact indexing equals `sum(len(c) for c in contents)`.

This means the scalar budget naturally distributes across contents — each child
draws from the `CountdownDrawer` independently, and the union length is the total
of all child lengths.

### Nesting constraint tracking

UnionArray introduces the first nesting constraints that need to be tracked during
tree building:

1. **UnionArray cannot be nested inside option or indexed nodes** — irrelevant for
   now (no option/indexed support yet), but will matter later.
2. **UnionArray cannot contain union-type children** — a `_build()` call producing
   a child for UnionArray must not itself produce a UnionArray at the top level.

The "no nested unions" constraint applies to **direct** children only. Indirect
nesting like `Union → Record → Union` or `Union → List → Union` is valid because
the intermediate node (Record or List) is the direct child, not another Union.

However, the tree builder can produce direct nesting: if `_build(depth+1)` returns
a UnionArray (because at that level, 2+ children were collected and `'union'` was
drawn), that UnionArray becomes a direct child of the parent. If the parent also
draws `'union'`, the result is a nested union, which the constructor rejects.

**Solution:** Pass an `allow_union_child` flag to `_build()`. When the parent is
UnionArray, call `_build(depth+1, allow_union_child=False)`. Inside `_build()`,
when `allow_union_child=False`, exclude `'union'` from the multi-child node types
(forcing RecordArray for 2+ children, which is always valid). Indirect nesting
remains possible because the flag is not propagated further down.

## Scalar Budget Considerations

With compact indexing, the union's total length is the sum of its children's
lengths. The `CountdownDrawer` already handles this — each child draws from the
shared budget independently. The union doesn't add extra scalars beyond what its
children consume.

This differs from RecordArray, where all children share the same record length
(each child has the same number of elements). In UnionArray, children can have
different lengths.

## Parameters to Consider

For `union_array_contents()`:

- **`contents`** — Strategy or list of strategies for the child contents, or
  `None` for default. The three-form dispatch pattern (None / Strategy / Content)
  used by existing wrappers needs adaptation: UnionArray needs 2+ children, not
  just one.
- **`max_contents`** — Maximum number of alternative contents. Default could be
  ~4. Capped at 128 (int8 limit), but practically much lower.

For `contents()` / `arrays()`:

- **`allow_union`** — Enable UnionArray generation. Default `True`.

## Real-World UnionArray Patterns

### Heterogeneous collections (JSON-like data)

```python
# Mixed types from JSON parsing
data = ak.from_iter([1.1, "hello", [1, 2], 3.3])
# Type: 4 * union[float64, string, var * int64]
```

### Union of record types (particle physics)

Different particle types in the same array, each with different field schemas:

```python
# Electrons and muons in the same collection
union = ak.concatenate([
    ak.Array([{'pt': 1.0, 'eta': 0.5}]),       # electrons
    ak.Array([{'pt': 2.0, 'mass': 105.7}]),     # muons
])
# Type: 2 * union[{pt: float64, eta: float64}, {pt: float64, mass: float64}]
```

Can be flattened with `ak.merge_union_of_records`:

```python
merged = ak.merge_union_of_records(union)
# Type: 2 * {pt: ?float64, eta: ?float64, mass: ?float64}
```

### `ak.concatenate` with incompatible schemas

```python
combined = ak.concatenate([
    ak.Array([1.0, 2.0]),
    ak.Array([[1, 2], [3]]),
])
# Type: 4 * union[float64, var * int64]
```

### Arrow interop

UnionArray corresponds to Apache Arrow's **dense union** type. Sparse Arrow
unions gain an index on conversion to Awkward.

## Open Questions

1. **`union_array_contents()` API**: The existing wrapper strategies take a single
   `content` parameter. UnionArray needs 2+ children. Options:
   - Accept `contents: list[st.SearchStrategy[Content]]` (list of strategies)
   - Accept a single strategy and draw multiple times
   - Generate children internally (like the tree builder does)

2. **Should contents be required to have distinct types?** Real-world unions
   typically have heterogeneous types. But `union[int64, int64]` is technically
   valid. For simplicity, we could allow homogeneous unions.

3. **Compact vs. sparse indexing?** Compact is simpler and exercises every content
   element. Sparse could test more edge cases. Start with compact.

4. **How does depth counting work?** RecordArray consumes a depth level. Should
   UnionArray also consume a depth level? It adds structural complexity but
   doesn't add "list nesting" in the same way. For consistency with RecordArray,
   it should consume a depth level.

5. **"Another edge?" minimum for UnionArray**: The tree builder's "another edge?"
   loop starts with 1 child and adds more. UnionArray requires >= 2. If only 1
   child is collected and UnionArray would be drawn, we need to either:
   - Force another child (draw one more `_build()` call)
   - Exclude UnionArray from the 1-child case (already handled by the current
     algorithm — 1 child uses `single_child_types`, 2+ uses multi-child types)

   The current algorithm already handles this: UnionArray is only in the
   multi-child pool, and multi-child is only reached when 2+ children exist.

6. **`allow_union` default**: Should it default to `True` or `False`? RecordArray
   defaults to `True`. For consistency, UnionArray should also default to `True`.
   However, UnionArray is less common in practice and adds nesting constraints.
   Start with `True` for consistency.

## Sources

- [ak.contents.UnionArray reference](https://awkward-array.org/doc/main/reference/generated/ak.contents.UnionArray.html)
- [How to create arrays — Direct constructors](https://awkward-array.org/doc/main/user-guide/how-to-create-constructors.html)
- [Direct constructors research](./2026-02-04-direct-constructors-research.md)
- [Record array research](./2026-02-17-record-array-research.md)
- [Bottom-up tree builder](../impl/2026-02-17-contents-tree-builder.md)
