# String and Bytestring Research

Date: 2026-02-12

## Overview

Awkward Array has **no dedicated StringArray class**. Strings and bytestrings
are variable-length lists of `uint8` bytes differentiated by `__array__`
parameters on both the outer list node and the inner `NumpyArray` node.

## Internal Representation

### Layout Structure

**String:**

```text
ListOffsetArray(parameters={"__array__": "string"})
  └── NumpyArray(dtype=uint8, parameters={"__array__": "char"})
```

**Bytestring:**

```text
ListOffsetArray(parameters={"__array__": "bytestring"})
  └── NumpyArray(dtype=uint8, parameters={"__array__": "byte"})
```

### Parameter Pairing

| Outer `__array__` | Inner `__array__` | Inner dtype | Python type |
| ----------------- | ----------------- | ----------- | ----------- |
| `"string"`        | `"char"`          | `uint8`     | `str`       |
| `"bytestring"`    | `"byte"`          | `uint8`     | `bytes`     |

Both parameters are **required** and must match. Mismatches raise `ValueError`
at construction time.

### Allowed Outer List Types

All three list-type Content nodes can wrap strings:

- **`ListOffsetArray`** -- most common; what `ak.Array(["..."])` produces.
  Single `offsets` array.
- **`ListArray`** -- separate `starts`/`stops` arrays. Can have unreachable
  content and non-contiguous slices.
- **`RegularArray`** -- for fixed-size strings (all same byte length). Produces
  type like `string[5]`.

## Constructor Examples

### `ListOffsetArray` String

```python
content = ak.contents.NumpyArray(
    np.array([104, 101, 108, 108, 111, 119, 111, 114, 108, 100], dtype=np.uint8),
    parameters={'__array__': 'char'},
)
offsets = ak.index.Index64(np.array([0, 5, 10, 10], dtype=np.int64))
layout = ak.contents.ListOffsetArray(
    offsets, content, parameters={'__array__': 'string'}
)
# ['hello', 'world', '']
```

### `ListOffsetArray` Bytestring

```python
content = ak.contents.NumpyArray(
    np.array([104, 101, 108, 108, 111], dtype=np.uint8),
    parameters={'__array__': 'byte'},
)
offsets = ak.index.Index64(np.array([0, 5], dtype=np.int64))
layout = ak.contents.ListOffsetArray(
    offsets, content, parameters={'__array__': 'bytestring'}
)
# [b'hello']
```

### `ListArray` String

```python
content = ak.contents.NumpyArray(
    np.array([104, 101, 108, 108, 111, 119, 111, 114, 108, 100], dtype=np.uint8),
    parameters={'__array__': 'char'},
)
starts = ak.index.Index64(np.array([0, 5], dtype=np.int64))
stops = ak.index.Index64(np.array([5, 10], dtype=np.int64))
layout = ak.contents.ListArray(
    starts, stops, content, parameters={'__array__': 'string'}
)
# ['hello', 'world']
```

### `RegularArray` String (Fixed-Width)

```python
content = ak.contents.NumpyArray(
    np.frombuffer(b'helloworld', dtype=np.uint8),
    parameters={'__array__': 'char'},
)
layout = ak.contents.RegularArray(content, size=5, parameters={'__array__': 'string'})
# ['hello', 'world']  -- type: 2 * string[5]
```

### Empty String Array (0 Elements, Properly Typed)

```python
content = ak.contents.NumpyArray(
    np.array([], dtype=np.uint8),
    parameters={'__array__': 'char'},
)
offsets = ak.index.Index64(np.array([0], dtype=np.int64))
layout = ak.contents.ListOffsetArray(
    offsets, content, parameters={'__array__': 'string'}
)
# []  -- type: 0 * string
```

Note: `ak.Array([])` produces `EmptyArray` with type `0 * unknown`, not a
string array. `EmptyArray` cannot hold parameters.

## Validation Constraints

### At Construction Time (Immediate)

1. **`NumpyArray.__init__`**: If `__array__` is `"char"` or `"byte"`, then
   `data.dtype` must be `np.uint8` and data must be 1-dimensional.

2. **`ListOffsetArray.__init__`** / **`ListArray.__init__`**: If `__array__` is
   `"string"`, then `content` must satisfy `content.is_numpy` and
   `content.parameter("__array__") == "char"`. Same for `"bytestring"` /
   `"byte"`.

3. **`RegularArray.__init__`**: Same validation as list types.

### Deferred Validation (`ak.is_valid()`)

- `offsets` / `starts` / `stops` must not reference beyond `len(content)`.
- `stops[i] >= starts[i]` for all `i`.

### UTF-8

Awkward **does not validate** that string content is valid UTF-8 at construction
time. It assumes UTF-8 encoding. On `to_list()`, bytes are decoded with
`errors="surrogateescape"`.

## Type and Form Representation

### Type

```python
# String
ak.types.ListType(
    ak.types.NumpyType('uint8', parameters={'__array__': 'char'}),
    parameters={'__array__': 'string'},
)
# Displays as: string

# Bytestring
ak.types.ListType(
    ak.types.NumpyType('uint8', parameters={'__array__': 'byte'}),
    parameters={'__array__': 'bytestring'},
)
# Displays as: bytes
```

### Form

```python
# String (ListOffsetForm)
ak.forms.ListOffsetForm(
    'i64',
    ak.forms.NumpyForm('uint8', parameters={'__array__': 'char'}),
    parameters={'__array__': 'string'},
)

# String (ListForm)
ak.forms.ListForm(
    'i64', 'i64',
    ak.forms.NumpyForm('uint8', parameters={'__array__': 'char'}),
    parameters={'__array__': 'string'},
)

# String (RegularForm) -- fixed-width
ak.forms.RegularForm(
    ak.forms.NumpyForm('uint8', parameters={'__array__': 'char'}),
    size=5,
    parameters={'__array__': 'string'},
)
```

## Nesting and Composition

### Nested Strings (List of Strings)

`__array__` parameters are only on the **innermost** list-type + NumpyArray pair.
Outer nesting layers have no `__array__` parameter.

```text
ListOffsetArray (no __array__)           -- outer list-of-lists
  ListOffsetArray (__array__="string")   -- the string list
    NumpyArray uint8 (__array__="char")  -- the bytes
```

### Nullable Strings (Option Wrapping)

Option nodes wrap the string `ListOffsetArray`. The `__array__` parameters
remain on the inner structure only.

```text
IndexedOptionArray (no __array__)
  ListOffsetArray (__array__="string")
    NumpyArray uint8 (__array__="char")
```

Type: `?string`

### Records with String Fields

```text
RecordArray
  field 'name': ListOffsetArray (__array__="string")
                  NumpyArray uint8 (__array__="char")
  field 'id':   NumpyArray int64
```

## Real-World Use Cases

### Bioinformatics (scverse ecosystem)

- **anndata/scirpy**: Store immune receptor sequences (TCR/BCR) as Awkward
  string arrays. AIRR data has string fields for gene names, junction sequences,
  and cell IDs.
- Pattern: Records with mixed string and numeric fields, nullable strings.

### High-Energy Physics (scikit-hep ecosystem)

- **uproot**: Reads ROOT `std::string` branches as Awkward string arrays.
- **coffea**: Processes HEP event data with string labels.
- Pattern: String fields in deeply nested records, string-based selections.

### Data Analysis

- **akimbo**: Pandas extension type using Awkward. Provides `.ak.str` accessor
  for string operations (split, join, contains, etc.).
- **dask-awkward**: Reads text files line-by-line as string arrays.
- Pattern: Bulk text processing, string splitting/matching.

### Common Patterns

1. **String fields in records** (most common): Records with one or more string
   fields alongside numeric data.
2. **Filtering by string content**: `ak.str.starts_with()`,
   `ak.str.match_substring()`, `ak.str.is_in()`.
3. **String transformation**: `ak.str.split_pattern()`, `ak.str.replace_substring()`.
4. **Variable-length strings** (dominant): Almost all use cases have
   variable-length strings, not fixed-width.
5. **Nullable strings**: Common in real data (missing values).

## Current Status in hypothesis-awkward

- **No string/bytestring strategy exists** in the codebase.
- Listed as "next steps" item 4 in
  `.design/api/2026-02-12-contents-api.md` (line 642).
- Planned `allow_string`/`allow_bytestring` flags in
  `.design/api/2026-01-21-types-api.md`.
- Existing `list_offset_array_contents()` and `list_array_contents()` do not
  accept `parameters` arguments.

## Strategy Design Implications

### Generation Approach

For **strings**, use `st.text()` and encode to UTF-8 to get valid bytes, then
construct `ListOffsetArray`/`ListArray` with proper parameters. This guarantees
valid UTF-8 without manual byte-level generation.

For **bytestrings**, use `st.binary()` directly since any bytes are valid.

### Integration Points

1. **New leaf-level strategies**: `string_contents()`, `bytestring_contents()`
   that produce `ListOffsetArray` with the required `__array__` parameters.
2. **Integration with `contents()`**: Add `allow_string` and
   `allow_bytestring` flags to the top-level `contents()` strategy.
3. **Integration with `arrays()`**: Propagate flags through to `contents()`.

### Key Constraint

String/bytestring nodes are **leaf-like** in the nesting sense: their `content`
is always a `NumpyArray(uint8)` with `__array__` parameter. They cannot wrap
arbitrary content like regular `ListOffsetArray` does. This means they should be
treated as a special case in the content generation tree, similar to
`numpy_array_contents()` or `empty_array_contents()`.

## Sources

- [How to create arrays of strings](https://awkward-array.org/doc/main/user-guide/how-to-create-strings.html)
- [Direct constructors (fastest)](https://awkward-array.org/doc/main/user-guide/how-to-create-constructors.html)
- [Read strings from a binary stream](https://awkward-array.org/doc/main/user-guide/how-to-strings-read-binary.html)
- [ak.contents.ListOffsetArray](https://awkward-array.org/doc/main/reference/generated/ak.contents.ListOffsetArray.html)
- [ak.contents.ListArray](https://awkward-array.org/doc/main/reference/generated/ak.contents.ListArray.html)
- [How to examine an array's type](https://awkward-array.org/doc/main/user-guide/how-to-examine-type.html)
- [scverse/anndata](https://github.com/scverse/anndata)
- [intake/akimbo](https://github.com/intake/akimbo)
- [dask-contrib/dask-awkward](https://github.com/dask-contrib/dask-awkward)
