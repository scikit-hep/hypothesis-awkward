# RecordArray Research

Date: 2026-02-17

## Overview

`RecordArray` is a multi-content node in Awkward Array that represents
named-field records or unnamed tuples. Unlike list nodes (which wrap a single
child content), `RecordArray` combines zero or more child contents — one per
field — into a single structure where each element is a record.

## Constructor Signature

```python
RecordArray(
    contents: Iterable[Content],
    fields: Iterable[str] | None,
    length: int | type[unknown_length] | None = None,
    *,
    parameters=None,
    backend=None,
)
```

Parameters:

- **`contents`** — An iterable of `Content` subclasses (the field values). Can
  be empty (`[]`).
- **`fields`** — A list of field name strings, or `None`. When `None`, the
  record is a **tuple** (unnamed fields, accessed by index). When a list,
  `len(fields) == len(contents)` is required.
- **`length`** — Explicit length. When `None`, computed as `min(len(c) for c in
  contents)`. When `len(contents) == 0` and `length` is `None`, raises
  `TypeError` — length must be specified for empty records.
- **`parameters`** — Optional dict. The key `__record__` is special for
  behavior dispatch (e.g., `{'__record__': 'Point'}`).

## Construction Examples

### Named record

```python
r = ak.contents.RecordArray([
    ak.contents.NumpyArray(np.array([1, 2, 3])),
    ak.contents.NumpyArray(np.array([4.0, 5.0, 6.0])),
], fields=['x', 'y'])
# [{x: 1, y: 4}, {x: 2, y: 5}, {x: 3, y: 6}]
# Type: 3 * {x: int64, y: float64}
```

### Tuple (unnamed)

```python
t = ak.contents.RecordArray([
    ak.contents.NumpyArray(np.array([1, 2, 3])),
    ak.contents.NumpyArray(np.array([4.0, 5.0, 6.0])),
], fields=None)
# [(1, 4), (2, 5), (3, 6)]
# Type: 3 * (int64, float64)
```

### Empty record (no fields)

```python
e = ak.contents.RecordArray([], fields=[], length=3)
# [{}, {}, {}]
# Type: 3 * {}
```

### Empty tuple (no fields)

```python
e = ak.contents.RecordArray([], fields=None, length=3)
# [(), (), ()]
# Type: 3 * ()
```

## Key Properties

| Property     | Named record                        | Tuple (`fields=None`)       |
| ------------ | ----------------------------------- | --------------------------- |
| `contents`   | List of Content objects             | Same                        |
| `fields`     | List of strings, e.g. `['x', 'y']`  | Auto-generated `['0', '1']` |
| `is_tuple`   | `False`                             | `True`                      |
| `length`     | min of content lengths, or explicit | Same                        |
| `parameters` | `{}` by default                     | Same                        |

## Nesting Rules

`RecordArray` has **no nesting restrictions** — neither as a parent nor as a
child. It can appear anywhere in the content tree:

| Context                      | Example                                            |
| ---------------------------- | -------------------------------------------------- |
| Inside lists                 | `ListOffsetArray(offsets, RecordArray(...))`       |
| Inside options               | `ByteMaskedArray(mask, RecordArray(...))`          |
| Inside unions                | `UnionArray(tags, idx, [RecordArray(...), ...])`   |
| Inside other records (field) | `RecordArray([RecordArray([...], ...), ...], ...)` |

And `RecordArray` can contain any content type as field values:

- `NumpyArray`, `EmptyArray` (leaves)
- `ListOffsetArray`, `ListArray`, `RegularArray` (lists)
- `RecordArray` (nested records)
- Option nodes, `IndexedArray`, `UnionArray` (future)
- String/bytestring content (leaf-like `ListOffsetArray` with `__array__`)

(From the nesting constraint table in
[direct-constructors-research](./2026-02-04-direct-constructors-research.md#content-nesting-constraints).)

## Length Coordination

All field contents must have length >= the record's length.

- When `length=None`: computed as `min(len(c) for c in contents)`.
- When `length` is explicit: each content must have `len(content) >= length`.
  Shorter content raises `ValueError`.
- When `len(contents) == 0` and `length=None`: raises `TypeError`.

For strategy generation, the simplest approach is to ensure all field contents
have exactly the same length.

## Field Names

- Any string is valid as a field name.
- Duplicate field names are allowed at the constructor level (though semantically
  questionable).
- Empty strings and spaces are allowed.
- For tuples (`fields=None`), field names are auto-generated as `'0'`, `'1'`,
  etc.

## RecordArray vs. Existing Wrapper Nodes

RecordArray is fundamentally different from the existing wrapper nodes in the
`contents/` package:

| Aspect              | List wrappers (Regular, ListOffset, List) | RecordArray                |
| ------------------- | ----------------------------------------- | -------------------------- |
| Number of children  | 1 (`content`)                             | 0+ (`contents`)            |
| Role                | Nesting layer (adds depth)                | Branching (adds width)     |
| Length relationship | Wrapper length != content length          | All contents share length  |
| Scalar budget       | Single child shares budget                | Budget split across fields |
| Buffers             | Has buffers (offsets, starts/stops)       | No buffers                 |

The key difference: list wrappers are **one-to-one** (wrap one child), while
RecordArray is **one-to-many** (combines multiple children).

## Integration with the Wrappers Pattern

The current `contents()` strategy uses a wrappers pattern:

```text
leaf → wrapper₁ → wrapper₂ → ... → wrapperₙ
```

RecordArray does not fit neatly into this linear chain because it needs
multiple children. Two possible integration approaches:

### Approach A: RecordArray as a Branching Wrapper

When RecordArray is selected as a wrapper at depth level k, it takes the
content built so far as one field and generates additional fields:

```text
leaf → wrapper₁ → ... → RecordArray(field0=prev, field1=extra₁, ...) → wrapperₖ₊₁ → ...
```

The additional fields are drawn from the leaf strategy (sharing the scalar
budget). This keeps RecordArray in the wrappers loop.

**Advantages:**

- Minimal change to the existing `contents()` structure
- Naturally shares the scalar budget
- RecordArray can appear at any depth, wrapped by lists, etc.

**Disadvantages:**

- Only one field (the passed-in content) is "deep" — extra fields are leaf-only
  unless they independently recurse
- The passed-in content is always field 0, creating a structural bias

### Approach B: RecordArray as a Recursive Composition

RecordArray generates all its field contents independently via recursive calls
to a content generation function:

```text
RecordArray(
    field0 = contents(budget=b₀, max_depth=d-1),
    field1 = contents(budget=b₁, max_depth=d-1),
    ...
)
```

**Advantages:**

- Each field can be arbitrarily deep/complex
- No structural bias toward one field
- Cleaner separation of concerns

**Disadvantages:**

- Requires refactoring `contents()` to accept a shared budget tracker
- More complex budget splitting (how to divide budget across fields)
- Risk of generating very large structures (many fields × deep nesting)

### Approach C: Hybrid — Branching Wrapper with Leaf-Only Extra Fields

A pragmatic middle ground: RecordArray as a wrapper in the existing loop, with
extra fields drawn from `leaf_contents()` only (no recursive nesting for extra
fields). The passed-in content can be arbitrarily deep.

```text
leaf → Regular → RecordArray(field0=prev, field1=leaf₁, field2=leaf₂) → ListOffset
```

**Advantages:**

- Simple to implement with the current `contents()` pattern
- Keeps generated records small (extra fields are flat)
- The budget is easily managed: extra fields draw from the same
  `CountdownDrawer`
- One field is guaranteed to be deep (the passed-in content), which exercises
  the interesting nesting cases

**Disadvantages:**

- Extra fields are always leaves — no `{x: var * int64, y: var * float64}`
  patterns. Only one field gets nesting.

This limitation could be acceptable for an initial version. Users who need all
fields to be deeply nested can compose `record_array_contents()` directly with
custom content strategies.

## Scalar Budget Considerations

With multiple fields, the scalar budget must be shared. The `CountdownDrawer`
already supports multiple draws from a shared budget — each field's content
would be one draw.

**Budget splitting:**

- In Approach C (leaf-only extra fields), the passed-in content has already
  consumed part of the budget. Extra fields draw the remaining budget.
- In Approach B (recursive), the budget must be pre-split or dynamically
  allocated. Dynamic allocation (via `CountdownDrawer`) is simpler.

## Parameters to Consider

For `record_array_contents()`:

- **`content`** — The three-form dispatch (None / Strategy / Content), same as
  existing wrappers. This becomes one field of the record.
- **`max_fields`** — Maximum number of fields (including the passed-in
  content). Default could be ~5.
- **`allow_tuple`** — Whether to generate tuple records (`fields=None`).
  Default `True`.

For `contents()` / `arrays()`:

- **`allow_record`** — Enable RecordArray generation. Default `True`.
- **`max_fields`** — Could be forwarded from top-level, or kept internal.

## Real-World RecordArray Patterns

From the scikit-HEP and scverse ecosystems:

### Records with mixed field types

Most common pattern: records with scalar, list, and string fields.

```python
# Particle data: record with scalar and list fields
RecordArray([
    NumpyArray(float64),       # pt (scalar)
    NumpyArray(float64),       # eta (scalar)
    ListOffsetArray(offsets,   # jets (variable-length list)
        NumpyArray(float64)),
], fields=['pt', 'eta', 'jets'])
```

### Nested records

Records containing other records (hierarchical data).

```python
# Event → particle records
RecordArray([
    RecordArray([             # muon fields
        NumpyArray(float64),  # pt
        NumpyArray(float64),  # eta
    ], fields=['pt', 'eta']),
    NumpyArray(int64),        # event_id
], fields=['muon', 'event_id'])
```

### Records inside lists

List of records — the table-like pattern.

```python
# Variable number of particles per event
ListOffsetArray(offsets,
    RecordArray([
        NumpyArray(float64),  # pt
        NumpyArray(float64),  # eta
    ], fields=['pt', 'eta'])
)
```

### Records with string fields

Records mixing strings and numeric data.

```python
# Bioinformatics: sequence records
RecordArray([
    string_content,           # gene_name (string)
    NumpyArray(float64),      # expression (scalar)
], fields=['gene_name', 'expression'])
```

## Open Questions

1. **Which integration approach?** Approach C (hybrid) seems like the best
   starting point — it's simple, integrates with the existing wrappers pattern,
   and can be expanded later.

2. **Should extra fields be leaf-only or allow nesting?** Leaf-only is simpler
   and avoids budget explosion. Nesting support can be added later.

3. **Empty records (0 fields)?** These are valid (`RecordArray([], [], length=n)`)
   but unusual. Should they be generated? Probably gated by `min_size == 0` or
   similar.

4. **Tuple vs. named?** Both should be supported. The ratio could be controlled
   by a parameter or left to Hypothesis's random choices.

5. **Field name generation?** Simple names like `f0`, `f1`, `f2` are sufficient
   for testing. Alternatively, use Hypothesis's `st.text()` with a restricted
   alphabet. Duplicate names should probably be avoided.

6. **How does `max_depth` interact with RecordArray?** RecordArray itself adds
   structural complexity (width) but arguably does not add "depth" in the same
   sense as list wrappers. It could be treated as not consuming a depth level,
   similar to how strings don't count toward depth.

## Sources

- [How to create arrays — Direct constructors](https://awkward-array.org/doc/main/user-guide/how-to-create-constructors.html)
- [ak.contents.RecordArray reference](https://awkward-array.org/doc/main/reference/generated/ak.contents.RecordArray.html)
- [Direct constructors research](./2026-02-04-direct-constructors-research.md)
- [String/bytestring research](./2026-02-12-string-bytestring-research.md)
