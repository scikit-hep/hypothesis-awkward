# API Design: `arrays()` Strategy

**Date:** 2026-02-04
**Status:** Implemented (initial version)
**Author:** Claude (with developer collaboration)

## Overview

This document describes the API for the `arrays()` strategy, which generates
`ak.Array` objects via direct Content constructors. The current implementation
generates arrays with `NumpyArray` leaf contents nested in `RegularArray`,
`ListOffsetArray`, and `ListArray` wrappers. The design anticipates progressive
addition of `RecordArray`, option types, unions, and more in later iterations.

## Background

From the [UX research](./../research/2026-01-21-ux-interview-01.md),
[type system research](./../research/2026-01-21-type-system-research.md), and
[direct constructors research](./../research/2026-02-04-direct-constructors-research.md):

- The ultimate goal is an `arrays()` strategy that generates fully general
  Awkward Arrays with multiple options to control the layout, data types, missing
  values, masks, and other array attributes. `type` and `form` parameters are in
  scope but not mandatory
- Direct constructors produce real arrays in a single step (no roundtrip through
  forms or buffers)
- Constructor validation catches invalid nesting at construction time
- `ak.Array(layout)` wrapping is zero-cost
- Target users: Awkward developers, scikit-HEP tool developers, physicists

### Why Start with Direct Constructors

Three possible generation pipelines were considered:

1. **Type -> Form -> buffers -> array** (most abstract, most roundtrips)
2. **Form -> buffers -> array** (via `ak.from_buffers()`)
3. **Content layout -> array** (direct constructors, single step)

Pipeline 3 is the most direct: each Content constructor validates its inputs, so
the strategy can lean on built-in checks. It also exercises more internal code
paths than the canonical representations produced by `from_type()`.

### Why Start with NumpyArray Only

- `NumpyArray` is the leaf node -- every Awkward Array tree terminates in one
- The existing `numpy_arrays()` strategy already generates NumPy ndarrays;
  `arrays()` reuses this work
- Starting with a single node type validates the `arrays()` interface before
  adding recursive complexity
- Users get immediate value: `arrays()` with no arguments generates flat arrays,
  which is the most common case

## Design Goals

1. **Incremental extensibility**: Adding new node types must not break the
   existing interface
2. **Reuse existing strategies**: Build on `numpy_arrays()`, `supported_dtypes()`,
   etc.
3. **Familiar patterns**: Follow the parameter conventions established by
   `numpy_arrays()`, `types()`, and `numpy_forms()`
4. **Sensible defaults**: Works well out of the box; generates interesting arrays
   without configuration
5. **Composition with types/forms (future)**: The API should accommodate `type`
   and `form` parameters when those pipelines are connected

## API

### Main Strategy: `arrays()`

```python
@st.composite
def arrays(
    draw: st.DrawFn,

    # --- Leaf data control ---
    dtypes: st.SearchStrategy[np.dtype] | None = None,
    allow_nan: bool = False,

    # --- Size control ---
    max_size: int = 10,

    # --- Node type control ---
    allow_regular: bool = True,
    allow_list_offset: bool = True,
    allow_list: bool = True,

    # --- Nesting control ---
    max_depth: int = 5,
) -> ak.Array:
```

### Parameter Details

#### `dtypes`

Strategy for NumPy dtypes used in `NumpyArray` leaf nodes.

- If `None` (default), uses `supported_dtypes()`.
- If a strategy, draws from it.
- Follows the same convention as `numpy_types(dtypes=...)` and
  `numpy_forms(dtypes=...)`.

Note: Unlike the existing `numpy_arrays()` strategy, `arrays()` does not accept
a plain `np.dtype` value -- only a strategy or `None`. This is because
`arrays()` may draw multiple dtypes for different leaf nodes in a compound
structure (e.g., different record fields). Accepting a single dtype as a
convenience shorthand can be added later without breaking the interface.

#### `allow_nan`

Generate potentially `NaN`/`NaT` values for relevant dtypes.

- `False` (default): no `NaN`/`NaT` values.
- `True`: `NaN`/`NaT` values may appear.
- Same semantics as `numpy_arrays(allow_nan=...)`.

#### `max_size`

Maximum total number of scalar values in the generated array.

- Default: `10`.
- Controls the total scalar budget across all leaf `NumpyArray` nodes, not just
  the outermost dimension length.
- Uses a "budgeted leaf" approach: each leaf draws up to the remaining budget,
  preventing compound arrays from growing unboundedly.

#### `allow_regular`, `allow_list_offset`, `allow_list`

Control which structural Content node types are enabled.

- All default to `True`.
- `allow_regular`: Generate `RegularArray` wrappers (fixed-size lists).
- `allow_list_offset`: Generate `ListOffsetArray` wrappers (variable-length lists
  via offsets).
- `allow_list`: Generate `ListArray` wrappers (variable-length lists via
  starts/stops).
- When all three are `False`, only flat `NumpyArray`-backed arrays are generated.

Future node type flags (not yet implemented): `allow_record`, `allow_option`,
`allow_union`, `allow_string`.

#### `max_depth`

Maximum nesting depth for structural wrappers.

- Default: `5`.
- `max_depth=0` forces leaf-only arrays (flat `NumpyArray`).
- The strategy draws a random depth between 0 and `max_depth`, then applies that
  many randomly chosen wrappers (`RegularArray`, `ListOffsetArray`, `ListArray`).

### Return Type

`arrays()` returns `st.SearchStrategy[ak.Array]`.

**Decision:** Return `ak.Array`, not the raw `Content` layout.

**Rationale:**

- `ak.Array` is the user-facing type. Users write code against `ak.Array`, so
  the strategy should produce what they consume.
- `ak.Array(layout)` is zero-cost wrapping.
- Users needing the raw layout can access it via `result.layout`.
- Matches the precedent set by `from_numpy()`, which returns `ak.Array`.

### Parameters NOT Included

#### `allow_structured` / `allow_inner_shape`

Not included. In `numpy_arrays()`, `allow_structured=True` generates structured
NumPy arrays (with named fields), which `ak.from_numpy` converts to
`RecordArray`. In `arrays()`, records will be generated natively via
`RecordArray` when `allow_record=True`. Similarly, multi-dimensional NumpyArrays
(inner_shape) will be covered by `RegularArray` wrapping.

The `arrays()` strategy generates 1-D `NumpyArray` nodes only. Higher-dimensional
structure comes from nesting (lists, records), not from multi-dimensional NumPy
arrays.

#### `type` / `form`

Not included in the initial version. These will be added when the Type->Array
and Form->Array pipelines are connected:

```python
# Future signature extension:
def arrays(
    ...,
    type: ak.types.Type | st.SearchStrategy[ak.types.Type] | None = None,
    form: ak.forms.Form | st.SearchStrategy[ak.forms.Form] | None = None,
) -> st.SearchStrategy[ak.Array]:
```

When `type` or `form` is given, the strategy would generate arrays matching that
type or form. This can be added without breaking the current interface.

#### `allow_datetime`

Not included as a separate parameter. Users who want to exclude datetime dtypes
can filter via the `dtypes` parameter:

```python
non_datetime = st_ak.supported_dtypes().filter(lambda d: d.kind not in ('M', 'm'))
st_ak.constructors.arrays(dtypes=non_datetime)
```

This keeps the `arrays()` parameter list focused on structural concerns. The
`dtypes` parameter already provides this control.

## Implementation

### Wrappers Pattern

The actual implementation uses a "wrappers" pattern rather than a recursive
`_contents()` strategy. The approach:

1. **Leaf strategy**: `_numpy_leaf()` generates a 1-D `NumpyArray` Content via
   `numpy_arrays()` with `allow_structured=False` and `max_dims=1`.
2. **Budgeted leaf**: `_budgeted_leaf()` wraps the leaf strategy with a scalar
   budget (closure over `remaining` counter) so the total number of scalars
   across all leaves stays within `max_size`.
3. **Wrapper functions**: `_wrap_regular()`, `_wrap_list_offset()`, and
   `_wrap_list()` each take a child Content strategy and wrap the drawn child in
   the corresponding structural node.
4. **Composition**: The main `arrays()` draws a random depth (0 to `max_depth`),
   chooses a wrapper for each level, draws a leaf, then applies wrappers from
   innermost to outermost.

```python
@st.composite
def arrays(draw, dtypes=None, max_size=10, allow_nan=False,
           allow_regular=True, allow_list_offset=True,
           allow_list=True, max_depth=5) -> ak.Array:
    wrappers = []
    if allow_regular:
        wrappers.append(_wrap_regular)
    if allow_list_offset:
        wrappers.append(_wrap_list_offset)
    if allow_list:
        wrappers.append(_wrap_list)

    if not wrappers or max_size == 0:
        layout = draw(_numpy_leaf(dtypes, allow_nan, max_size))
    else:
        leaf_st = _budgeted_leaf(dtypes, allow_nan, max_size)
        depth = draw(st.integers(min_value=0, max_value=max_depth))
        chosen_wrappers = [draw(st.sampled_from(wrappers)) for _ in range(depth)]
        layout = draw(leaf_st)
        for wrapper in reversed(chosen_wrappers):
            layout = draw(wrapper(st.just(layout)))

    return ak.Array(layout)
```

### Future Expansion

When adding `RecordArray`, option types, and unions, the wrappers pattern will
need to evolve to handle content nesting constraints (e.g., option nodes cannot
wrap other option nodes). This may require switching to the `_contents()`
recursive strategy sketched in the original design, or adding a `_forbidden`
parameter to wrappers.

## Internal Strategies

### `_numpy_leaf()`

Generates a 1-D `NumpyArray` Content via `numpy_arrays()`:

```python
def _numpy_leaf(dtypes, allow_nan, max_size) -> st.SearchStrategy[ak.contents.NumpyArray]:
    return st_ak.numpy_arrays(
        dtype=dtypes, allow_structured=False,
        allow_nan=allow_nan, max_dims=1, max_size=max_size,
    ).map(ak.contents.NumpyArray)
```

### `_budgeted_leaf()`

Wraps `_numpy_leaf()` with a scalar budget via a closure. Each draw decrements
`remaining`; when the budget is exhausted, raises `_BudgetExhausted`.

### `_wrap_regular()`, `_wrap_list_offset()`, `_wrap_list()`

Each takes a child Content strategy, draws the child, and wraps it in the
corresponding structural node with randomly generated parameters (offsets, size,
etc.).

### Relationship to Existing Strategies

```text
Existing:
  supported_dtypes()  -->  numpy_arrays()   -->  from_numpy()  --> ak.Array
                       -->  numpy_types()                        (via ak.from_numpy)
                       -->  numpy_forms()

New:
  supported_dtypes()  -->  numpy_arrays()  -->  arrays()  --> ak.Array
                                                              (via Content constructors)
```

`arrays()` delegates to `numpy_arrays()` with `allow_structured=False` and
`max_dims=1` to obtain a 1-D simple NumPy array, then wraps it in
`ak.contents.NumpyArray` directly. This reuses the existing dtype handling and
empty-array generation logic in `numpy_arrays()`.

## Module Location

### Directory Structure

```text
src/hypothesis_awkward/strategies/
+-- constructors/
|   +-- __init__.py           # Re-exports arrays()
|   +-- arrays_.py            # arrays() strategy and internal helpers
+-- builtins_/
+-- forms/
+-- misc/
+-- numpy/
+-- pandas/
+-- types/
+-- __init__.py               # Re-exports arrays via constructors namespace
```

All internal helpers (`_numpy_leaf`, `_budgeted_leaf`, `_wrap_regular`,
`_wrap_list_offset`, `_wrap_list`) live in `arrays_.py`. As more node types are
added, they may be split into separate files:

```text
src/hypothesis_awkward/strategies/constructors/
+-- __init__.py
+-- arrays_.py                # Main arrays() strategy + current helpers
+-- record.py                 # _record_array_contents() (future)
+-- option.py                 # _option_array_contents() (future)
+-- union.py                  # _union_array_contents() (future)
+-- string.py                 # _string_array_contents() (future)
```

### Public API

The `constructors` subpackage is available as a namespace under `strategies/`:

```python
import hypothesis_awkward.strategies as st_ak

st_ak.constructors.arrays()
```

Not re-exported at the `st_ak` top level. The `constructors` module is imported
as a namespace (`from . import constructors`), so it is accessed as
`st_ak.constructors.arrays()`.

## Design Decisions

### 1. Implemented `allow_*` Flags Default to `True`

**Decision:** `allow_regular`, `allow_list_offset`, and `allow_list` default to
`True`. Future flags for unimplemented node types will be added when those node
types are implemented.

**Rationale:**

- Maximizes coverage by default -- `arrays()` with no arguments generates a
  variety of structural layouts.
- Users who want flat arrays can set all flags to `False`.
- New `allow_*` flags will be added (defaulting to `True`) as new node types are
  implemented.

### 2. `max_size` (Total Scalar Budget)

**Decision:** Use `max_size` to control the total number of scalar values across
all leaf `NumpyArray` nodes.

**Rationale:**

- In nested structures, controlling the outermost dimension length alone does
  not bound total array size.
- A scalar budget provides a natural cap that works regardless of nesting depth.
- The "budgeted leaf" approach (closure with `remaining` counter) ensures the
  budget is respected across multiple leaf draws.

### 3. Return `ak.Array`, Not `Content`

**Decision:** Return `ak.Array`.

**Rationale:**

- `ak.Array` is the user-facing type. Tests are written against `ak.Array`.
- `ak.Array(layout)` is zero-cost (no data copy).
- Raw layout is accessible via `result.layout`.
- Matches `from_numpy()` precedent.

### 4. No `allow_structured` Parameter

**Decision:** Omit `allow_structured`. Structured arrays (records) will come
from `allow_record=True`.

**Rationale:**

- In the `numpy_arrays()` -> `from_numpy()` path, structured NumPy dtypes
  produce RecordArray layouts. This was a side-effect of using `ak.from_numpy`.
- In `arrays()`, RecordArray is a first-class node type controlled by
  `allow_record`.
- Mixing structured dtypes with direct constructors would complicate the
  implementation without adding value.

### 5. `dtypes` Accepts Only Strategy or None, Not Plain `np.dtype`

**Decision:** `dtypes` accepts `st.SearchStrategy[np.dtype] | None`, not
`np.dtype | st.SearchStrategy[np.dtype] | None`.

**Rationale:**

- In compound arrays, multiple leaf nodes may need different dtypes.
- Accepting a plain `np.dtype` would fix all leaves to the same dtype, which is
  limiting.
- Users who want a single dtype can use `st.just(np.dtype('float64'))`.
- This can be relaxed later (accepting plain dtype as a convenience) without
  breaking the interface.

**Alternative considered:** Accepting plain `np.dtype` and wrapping in
`st.just()`, as `numpy_arrays(dtype=...)` does. Deferred to keep the initial
interface strict and revisit based on user feedback.

### 6. Separate `constructors/` Directory

**Decision:** Place `arrays()` in `strategies/constructors/`, not in
`strategies/` root or `strategies/numpy/`.

**Rationale:**

- The directory name reflects the approach (direct constructors), not the output.
  This leaves room for alternative approaches later (e.g., `builders/`,
  `from_types/`).
- `arrays()` is the main entry point, not specific to NumPy.
- It will grow to contain multiple internal modules (one per node type).
- Follows the pattern of `types/`, `forms/`, etc.
- `strategies/numpy/` is for the NumPy-specific path (`ak.from_numpy`).

### 7. Add `allow_*` Flags Only When Implemented

**Decision:** Only include `allow_*` flags for node types that are actually
implemented. Currently: `allow_regular`, `allow_list_offset`, `allow_list`.

**Rationale:**

- Avoids dead parameters that have no effect.
- The signature grows as features are added, which is natural for an
  experimental strategy.
- Keyword-only arguments make this a non-breaking change.

## Usage Examples

### Basic Usage

```python
import hypothesis_awkward.strategies as st_ak
from hypothesis import given

@given(a=st_ak.constructors.arrays())
def test_something(a):
    # a is an ak.Array, possibly nested with RegularArray/ListOffsetArray/ListArray
    assert isinstance(a, ak.Array)
```

### Specific Dtype

```python
import numpy as np
from hypothesis import strategies as st
from hypothesis_awkward.util import iter_numpy_arrays

@given(a=st_ak.constructors.arrays(dtypes=st.just(np.dtype('float64'))))
def test_float_arrays(a):
    for leaf in iter_numpy_arrays(a):
        assert leaf.dtype == np.dtype('float64')
```

### Integer Dtypes Only

```python
int_dtypes = st_ak.supported_dtypes().filter(lambda d: d.kind == 'i')

@given(a=st_ak.constructors.arrays(dtypes=int_dtypes))
def test_integer_arrays(a):
    for leaf in iter_numpy_arrays(a):
        assert leaf.dtype.kind == 'i'
```

### Allow NaN

```python
float_dtypes = st_ak.supported_dtypes().filter(lambda d: d.kind == 'f')

@given(a=st_ak.constructors.arrays(dtypes=float_dtypes, allow_nan=True))
def test_nan_handling(a):
    # Test code that must handle NaN correctly
    ...
```

### Control Size

```python
@given(a=st_ak.constructors.arrays(max_size=100))
def test_larger_arrays(a):
    # Total scalar count across all leaves is at most 100
    ...
```

### Flat Arrays Only

```python
@given(a=st_ak.constructors.arrays(
    allow_regular=False, allow_list_offset=False, allow_list=False,
))
def test_flat_arrays(a):
    assert isinstance(a.layout, ak.contents.NumpyArray)
```

### Future: With Type Constraint

```python
# Future: generate arrays matching a specific type
@given(a=st_ak.constructors.arrays(type=ak.types.ListType(ak.types.NumpyType('float64'))))
def test_typed_arrays(a):
    assert a.type.content == ak.types.NumpyType('float64')
```

## Testing Plan

Following the patterns in
[testing-patterns.md](./../../.claude/rules/testing-patterns.md):

Tests are implemented in `tests/strategies/constructors/test_arrays.py`.

### 1. TypedDict for kwargs

```python
class ArraysKwargs(TypedDict, total=False):
    '''Options for `arrays()` strategy.'''
    dtypes: st.SearchStrategy[np.dtype] | None
    max_size: int
    allow_nan: bool
    allow_regular: bool
    allow_list_offset: bool
    allow_list: bool
    max_depth: int
```

### 2. Strategy for kwargs with `st_ak.RecordDraws` and `st_ak.Opts`

```python
def arrays_kwargs() -> st.SearchStrategy[st_ak.Opts[ArraysKwargs]]:
    '''Strategy for options for `arrays()` strategy.'''
    return (
        st.fixed_dictionaries(
            {},
            optional={
                'dtypes': st.one_of(
                    st.none(),
                    st.just(st_ak.RecordDraws(st_ak.supported_dtypes())),
                ),
                'max_size': st.integers(min_value=0, max_value=50),
                'allow_nan': st.booleans(),
                'allow_regular': st.booleans(),
                'allow_list_offset': st.booleans(),
                'allow_list': st.booleans(),
                'max_depth': st.integers(min_value=0, max_value=5),
            },
        )
        .map(lambda d: cast(ArraysKwargs, d))
        .map(st_ak.Opts)
    )
```

### 3. Main property-based test

Verifies all options are respected: `max_size` (total scalars), per-type gating
(`allow_regular`, `allow_list_offset`, `allow_list`), dtypes via leaf arrays,
`allow_nan`, and `max_depth`.

### 4. Edge case reachability tests

Tests use `find()` to verify that the strategy can produce:

- Empty arrays
- Arrays using the full scalar budget
- Arrays with NaN (when allowed)
- Integer dtype arrays
- Arrays at exactly `max_depth`
- Nested arrays (depth >= 2)
- `RegularArray` with size=0
- `ListOffsetArray` and `ListArray`
- Variable-length sublists
- Empty sublists

### 5. File structure for tests

```text
tests/strategies/constructors/
+-- __init__.py
+-- test_arrays.py
```

## Alternatives Considered

### Alternative A: Extend `from_numpy()` Instead

Expand `from_numpy()` to support more node types.

**Rejected because:**

- `from_numpy()` is specifically about the `ak.from_numpy()` path.
- Direct constructors exercise a different code path.
- Mixing concerns would make `from_numpy()` confusing.
- A separate `arrays()` strategy has a clearer purpose.

### Alternative B: Start from Forms

Generate `Form` objects first, then use `ak.from_buffers()` to produce arrays.

**Rejected because:**

- Requires buffer generation with complex invariants (sorted offsets, etc.).
- Two-step process (form + buffers) is harder to debug.
- Direct constructors validate at construction time.
- Can be added later as an alternative pipeline if needed.

### Alternative C: Start from Types

Generate `Type` objects, convert to `Form`, then to arrays.

**Rejected because:**

- Three-step pipeline (type -> form -> buffers -> array).
- Most indirection, hardest to debug.
- One type maps to multiple forms; need to resolve ambiguity.
- Can be added later via the `type` parameter.

### Alternative D: `max_length` Instead of `max_size`

Control the outermost dimension length rather than total scalars.

**Rejected because:**

- In nested structures, outermost length alone does not bound total array size.
- A scalar budget (`max_size`) provides a natural cap regardless of nesting depth.
- The "budgeted leaf" approach made `max_size` straightforward to implement.

### Alternative E: Accept Plain `np.dtype` for `dtypes`

Accept `np.dtype | st.SearchStrategy[np.dtype] | None` as in `numpy_arrays()`.

**Deferred because:**

- In compound arrays, different leaves may need different dtypes.
- Accepting a plain dtype implies all leaves share it, which is limiting.
- Can be added later as a convenience without breaking changes.

## Open Questions

1. **Should `allow_empty` be a parameter?** Awkward Array has `EmptyArray` (a
   zero-length placeholder with `UnknownType`). Should `arrays()` be able to
   generate these? They are unusual and may cause issues in downstream code.
   Tentative answer: No, defer to a future `allow_empty` flag.

2. **Should `allow_indexed` be a parameter?** `IndexedArray` is type-transparent
   (it does not change the type of its content). It adds a level of indirection
   that exercises different code paths. This is valuable for testing Awkward
   internals but may be confusing for end users. Tentative answer: Yes, add it
   when implementing option types (since `IndexedOptionArray` is the main option
   representation and `IndexedArray` shares its machinery).

3. **When `type`/`form` parameters are added, how do they interact with
   `allow_*` flags?** Tentative answer: When `type` or `form` is given, it fully
   determines the structure and `allow_*` flags are ignored (similar to how
   `numpy_forms(type_=...)` ignores other parameters).

## Completed

1. ~~Implement `arrays()` with NumpyArray-only support~~ ✓
2. ~~Add `RegularArray`, `ListOffsetArray`, `ListArray` support~~ ✓
3. ~~Add scalar budget approach (`max_size`)~~ ✓
4. ~~Add `max_depth` parameter~~ ✓
5. ~~Add tests following the testing plan~~ ✓
6. ~~Export from `strategies/__init__.py`~~ ✓

## Next Steps

1. Add `RecordArray` support (`allow_record`)
2. Add option type support (`allow_option`) -- `IndexedOptionArray`,
   `ByteMaskedArray`, `BitMaskedArray`, `UnmaskedArray`
3. Add `UnionArray` support (`allow_union`)
4. Add string/bytestring support (`allow_string`)
5. Implement content nesting constraint enforcement
6. Consider connecting `type`/`form` parameters
