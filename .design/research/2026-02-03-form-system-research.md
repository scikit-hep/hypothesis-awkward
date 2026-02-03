# Awkward Array Form System Research

**Date:** 2026-02-03
**Purpose:** Inform the design of form-generating strategies for hypothesis-awkward

## Form Class Hierarchy

Awkward Array v2.8.11 has 12 Form subclasses (all `@final`) plus a base class:

```text
Form (base class)
+-- NumpyForm        - Primitive/multidimensional numeric data
+-- EmptyForm        - Zero-length placeholder (UnknownType)
+-- RegularForm      - Fixed-size lists
+-- ListOffsetForm   - Variable-length lists (single offsets buffer)
+-- ListForm         - Variable-length lists (starts + stops buffers)
+-- RecordForm       - Named-field records or unnamed tuples
+-- IndexedForm      - Re-indexed view (type-transparent)
+-- IndexedOptionForm - Nullable via index (-1 = missing)
+-- ByteMaskedForm   - Nullable via byte mask
+-- BitMaskedForm    - Nullable via bit mask
+-- UnmaskedForm     - Non-nullable option wrapper
+-- UnionForm        - Tagged union of multiple content types
```

**Location:** `awkward/forms/` directory, one file per class.

### Meta Mixin Hierarchy

Each Form also inherits a `Meta` mixin from `awkward/_meta/` providing boolean
flags:

| Flag          | True for                                                       |
| ------------- | -------------------------------------------------------------- |
| `is_numpy`    | NumpyForm                                                      |
| `is_unknown`  | EmptyForm                                                      |
| `is_regular`  | RegularForm                                                    |
| `is_list`     | ListOffsetForm, ListForm, RegularForm                          |
| `is_record`   | RecordForm                                                     |
| `is_indexed`  | IndexedForm, IndexedOptionForm                                 |
| `is_option`   | IndexedOptionForm, ByteMaskedForm, BitMaskedForm, UnmaskedForm |
| `is_union`    | UnionForm                                                      |
| `is_leaf`     | NumpyForm, EmptyForm                                           |
| `is_unmasked` | UnmaskedForm                                                   |

Note: `IndexedOptionForm` sets both `is_indexed = True` and `is_option = True`.
`UnmaskedForm` sets both `is_option = True` and `is_unmasked = True`.
`RegularForm` sets both `is_regular = True` and `is_list = True`.

## Form Subclasses

### NumpyForm

**Location:** `awkward/forms/numpyform.py:49-67`

```python
NumpyForm(primitive, inner_shape=(), *, parameters=None, form_key=None)
```

- **primitive** -- normalized via `dtype_to_primitive(primitive_to_dtype(primitive))`
  round-trip; raises `TypeError` for unrecognized primitives
- **inner_shape** -- tuple of positive integers for fixed-size sub-arrays;
  default `()` (most common)
- **Type:** `NumpyType`, wrapped in reversed `RegularType` layers for non-empty
  inner_shape
- **Content class:** `NumpyArray`
- **Buffers:** `data`
- **Serialization class:** `"NumpyArray"`

### EmptyForm

**Location:** `awkward/forms/emptyform.py`

```python
EmptyForm(*, parameters=None, form_key=None)
```

- **Cannot have parameters** -- raises `TypeError` if parameters is not None
- **Type:** `UnknownType`
- **Content class:** `EmptyArray`
- **Buffers:** none
- **Serialization class:** `"EmptyArray"`

### RegularForm

**Location:** `awkward/forms/regularform.py`

```python
RegularForm(content, size, *, parameters=None, form_key=None)
```

- **content** -- must be a Form subclass
- **size** -- non-negative integer (or `unknown_length` singleton)
- **Type:** `RegularType(content.type, size)`
- **Content class:** `RegularArray`
- **Buffers:** none (size is metadata, not a buffer)
- **Serialization class:** `"RegularArray"`

### ListOffsetForm

**Location:** `awkward/forms/listoffsetform.py`

```python
ListOffsetForm(offsets, content, *, parameters=None, form_key=None)
```

- **offsets** -- index type string (`i32`, `u32`, `i64`)
- **content** -- must be a Form subclass
- **Type:** `ListType(content.type)`
- **Content class:** `ListOffsetArray`
- **Buffers:** `offsets`
- **Serialization class:** `"ListOffsetArray"`
- **Note:** most common list form; default from `from_type()`

### ListForm

**Location:** `awkward/forms/listform.py`

```python
ListForm(starts, stops, content, *, parameters=None, form_key=None)
```

- **starts, stops** -- index type strings (can differ from each other)
- **content** -- must be a Form subclass
- **Type:** `ListType(content.type)`
- **Content class:** `ListArray`
- **Buffers:** `starts`, `stops`
- **Serialization class:** `"ListArray"`

### RecordForm

**Location:** `awkward/forms/recordform.py`

```python
RecordForm(contents, fields, *, parameters=None, form_key=None)
```

- **contents** -- iterable of Form subclasses
- **fields** -- list of field name strings, or `None` for tuple
- **Validation:** `len(fields) == len(contents)` when fields is not None
- **Type:** `RecordType(content_types, fields)`
- **Content class:** `RecordArray`
- **Buffers:** none
- **Serialization class:** `"RecordArray"`

### IndexedForm

**Location:** `awkward/forms/indexedform.py`

```python
IndexedForm(index, content, *, parameters=None, form_key=None)
```

- **index** -- index type string
- **content** -- must be a Form subclass
- **Type:** `content.type` with merged parameters (**type-transparent** -- no
  `IndexedType` exists)
- **Content class:** `IndexedArray`
- **Buffers:** `index`
- **Serialization class:** `"IndexedArray"`
- **`simplified()`** -- absorbs into union content; converts to
  `IndexedOptionForm` for option content; flattens nested indexed

### IndexedOptionForm

**Location:** `awkward/forms/indexedoptionform.py`

```python
IndexedOptionForm(index, content, *, parameters=None, form_key=None)
```

- **index** -- index type string; -1 values indicate missing
- **content** -- must be a Form subclass
- **Type:** `OptionType(content.type)` with `simplify_option_union()`
- **Content class:** `IndexedOptionArray`
- **Buffers:** `index`
- **Serialization class:** `"IndexedOptionArray"`
- **Note:** default option form from `from_type()`
- **`simplified()`** -- flattens nested option/indexed

### ByteMaskedForm

**Location:** `awkward/forms/bytemaskedform.py`

```python
ByteMaskedForm(mask, content, valid_when, *, parameters=None, form_key=None)
```

- **mask** -- index type string (typically `i8`)
- **content** -- must be a Form subclass
- **valid_when** -- bool; whether mask=1 means valid or invalid
- **Type:** `OptionType(content.type)` with `simplify_option_union()`
- **Content class:** `ByteMaskedArray`
- **Buffers:** `mask`
- **Serialization class:** `"ByteMaskedArray"`

### BitMaskedForm

**Location:** `awkward/forms/bitmaskedform.py`

```python
BitMaskedForm(mask, content, valid_when, lsb_order, *, parameters=None, form_key=None)
```

- **mask** -- index type string (typically `u8`)
- **content** -- must be a Form subclass
- **valid_when** -- bool; whether bit=1 means valid or invalid
- **lsb_order** -- bool; whether least-significant bit is first
- **Type:** `OptionType(content.type)` with `simplify_option_union()`
- **Content class:** `BitMaskedArray`
- **Buffers:** `mask` (ceil(N/8) bytes)
- **Serialization class:** `"BitMaskedArray"`

### UnmaskedForm

**Location:** `awkward/forms/unmaskedform.py`

```python
UnmaskedForm(content, *, parameters=None, form_key=None)
```

- **content** -- must be a Form subclass
- **Type:** `OptionType(content.type)` with `simplify_option_union()`
- **Content class:** `UnmaskedArray`
- **Buffers:** none
- **Serialization class:** `"UnmaskedArray"`
- **`simplified()`** -- absorbs indexed/option content

### UnionForm

**Location:** `awkward/forms/unionform.py`

```python
UnionForm(tags, index, contents, *, parameters=None, form_key=None)
```

- **tags** -- index type string (typically `i8`)
- **index** -- index type string (typically `i64`)
- **contents** -- iterable of Form subclasses
- **Type:** `UnionType([c.type for c in contents])`
- **Content class:** `UnionArray`
- **Buffers:** `tags`, `index`
- **Serialization class:** `"UnionArray"`
- **Equality:** uses permutation matching on contents

## Form to Type Mapping

One Form maps to exactly one Type (deterministic via `.type` property). One Type
can map to multiple Forms.

| Form              | Type          | Notes                                       |
| ----------------- | ------------- | ------------------------------------------- |
| NumpyForm         | NumpyType     | inner_shape wraps in RegularType layers     |
| EmptyForm         | UnknownType   |                                             |
| RegularForm       | RegularType   |                                             |
| ListOffsetForm    | ListType      | Same type as ListForm                       |
| ListForm          | ListType      | Same type as ListOffsetForm                 |
| RecordForm        | RecordType    |                                             |
| IndexedForm       | (transparent) | Returns content.type with merged parameters |
| IndexedOptionForm | OptionType    | With simplify_option_union()                |
| ByteMaskedForm    | OptionType    | With simplify_option_union()                |
| BitMaskedForm     | OptionType    | With simplify_option_union()                |
| UnmaskedForm      | OptionType    | With simplify_option_union()                |
| UnionForm         | UnionType     |                                             |

### Structural Mismatches

**IndexedForm is type-transparent.** There is no `IndexedType`. The `.type`
property returns `content.type` with merged parameters. This means
`IndexedForm('i64', NumpyForm('float64'))` has type `NumpyType('float64')` --
the indexing layer is invisible at the type level.

**NumpyForm inner_shape creates RegularType layers.** A
`NumpyForm('int32', (2, 3))` produces
`RegularType(RegularType(NumpyType('int32'), 3), 2)` -- nested RegularType
wrapping, not a single type.

**Multiple Forms map to the same Type:**

- `ListOffsetForm` and `ListForm` both produce `ListType`
- `IndexedOptionForm`, `ByteMaskedForm`, `BitMaskedForm`, and `UnmaskedForm` all
  produce `OptionType`

### Canonical Forms from `from_type()`

`from_type()` at `awkward/forms/form.py` picks one canonical Form per Type:

| Type        | Canonical Form                  |
| ----------- | ------------------------------- |
| ListType    | `ListOffsetForm('i64', ...)`    |
| RegularType | `RegularForm(..., size)`        |
| RecordType  | `RecordForm([...], fields)`     |
| OptionType  | `IndexedOptionForm('i64', ...)` |
| UnionType   | `UnionForm('i8', 'i64', [...])` |
| NumpyType   | `NumpyForm(primitive)`          |
| UnknownType | `EmptyForm()`                   |

This mapping is lossy -- `from_type()` always picks `ListOffsetForm` over
`ListForm`, always picks `IndexedOptionForm` over `ByteMaskedForm`, etc.

## Form to Content Mapping

There is a 1:1 correspondence between Form and Content (layout) classes:

| Form              | Content Class      |
| ----------------- | ------------------ |
| NumpyForm         | NumpyArray         |
| EmptyForm         | EmptyArray         |
| RegularForm       | RegularArray       |
| ListOffsetForm    | ListOffsetArray    |
| ListForm          | ListArray          |
| RecordForm        | RecordArray        |
| IndexedForm       | IndexedArray       |
| IndexedOptionForm | IndexedOptionArray |
| ByteMaskedForm    | ByteMaskedArray    |
| BitMaskedForm     | BitMaskedArray     |
| UnmaskedForm      | UnmaskedArray      |
| UnionForm         | UnionArray         |

Each Content class has `form_cls` pointing back to its Form class (e.g.,
`NumpyArray.form_cls = NumpyForm`). `ak.from_buffers()` uses a Form to
reconstitute the corresponding Content from buffer data.

## Shared Constructor Parameters

All Form constructors accept `parameters` and `form_key` as keyword-only
arguments. These are validated by `Form._init()`.

### parameters

- **Type:** `dict` or `None`
- **Default:** `None`
- **Exception:** `EmptyForm` raises `TypeError` if parameters is not None
- **Purpose:** metadata dict passed through to Content and Type objects

Recognized parameter keys:

| Key               | Purpose                                        | Relevant Forms                      |
| ----------------- | ---------------------------------------------- | ----------------------------------- |
| `__array__`       | Array kind (`"string"`, `"bytestring"`, etc.)  | ListOffsetForm/ListForm + NumpyForm |
| `__record__`      | Custom record class name for behavior dispatch | RecordForm                          |
| `__list__`        | Custom list class name for behavior dispatch   | List forms                          |
| `__categorical__` | Marks categorical arrays                       | IndexedForm, IndexedOptionForm      |

### form_key

- **Type:** `str` or `None`
- **Default:** `None`
- **Purpose:** identifier for buffer management in serialization/deserialization
- **Uniqueness:** not enforced, though convention suggests uniqueness
- **Recommended:** use `form.form_with_unique_keys()` for automatic assignment

`form_with_unique_keys()` post-processes a form tree, assigning unique keys like
`"node0"`, `"node1"`, etc.

### Index Type Strings

Forms that reference index/offset/mask buffers accept index type strings from
`index_to_dtype`:

```python
index_to_dtype = {
    'i8':  np.dtype(np.int8),
    'u8':  np.dtype(np.uint8),
    'i32': np.dtype(np.int32),
    'u32': np.dtype(np.uint32),
    'i64': np.dtype(np.int64),
}
```

All 5 types are accepted by every form parameter (offsets, starts, stops, index,
mask, tags). Conventional usage:

| Parameter | Typical type | Notes                                     |
| --------- | ------------ | ----------------------------------------- |
| offsets   | `i64`        | ListOffsetForm                            |
| starts    | `i64`        | ListForm                                  |
| stops     | `i64`        | ListForm                                  |
| index     | `i64`        | IndexedForm, IndexedOptionForm, UnionForm |
| mask      | `i8`         | ByteMaskedForm                            |
| mask      | `u8`         | BitMaskedForm                             |
| tags      | `i8`         | UnionForm                                 |

## Form Tree Structure

Forms compose recursively to describe nested data structures. Leaf forms
(`NumpyForm`, `EmptyForm`) have no children. Container forms reference one or
more child forms:

| Form              | Children                        |
| ----------------- | ------------------------------- |
| NumpyForm         | none (leaf)                     |
| EmptyForm         | none (leaf)                     |
| RegularForm       | `content` (1 child)             |
| ListOffsetForm    | `content` (1 child)             |
| ListForm          | `content` (1 child)             |
| RecordForm        | `contents` (0 or more children) |
| IndexedForm       | `content` (1 child)             |
| IndexedOptionForm | `content` (1 child)             |
| ByteMaskedForm    | `content` (1 child)             |
| BitMaskedForm     | `content` (1 child)             |
| UnmaskedForm      | `content` (1 child)             |
| UnionForm         | `contents` (1 or more children) |

### Example Form Trees

Simple jagged array (`var * float64`):

```text
ListOffsetForm('i64')
+-- NumpyForm('float64')
```

Record with optional field (`{x: float64, y: ?int32}`):

```text
RecordForm(fields=['x', 'y'])
+-- NumpyForm('float64')
+-- IndexedOptionForm('i64')
    +-- NumpyForm('int32')
```

Doubly jagged array (`var * var * int64`):

```text
ListOffsetForm('i64')
+-- ListOffsetForm('i64')
    +-- NumpyForm('int64')
```

### Special Patterns

**String:** variable-length list of `uint8` with `__array__` parameters:

```text
ListOffsetForm('i64', parameters={'__array__': 'string'})
+-- NumpyForm('uint8', parameters={'__array__': 'char'})
```

**Bytestring:** same structure with `"bytestring"` and `"byte"` parameters.

**Empty record (tuple with 0 fields):**

```text
RecordForm(contents=[], fields=None)
```

## Serialization

### `to_dict()` / `from_dict()`

Forms serialize to and deserialize from JSON-compatible dicts. The `"class"` key
identifies the Content class name (not the Form class name):

| Form              | `"class"` value        |
| ----------------- | ---------------------- |
| NumpyForm         | `"NumpyArray"`         |
| EmptyForm         | `"EmptyArray"`         |
| RegularForm       | `"RegularArray"`       |
| ListOffsetForm    | `"ListOffsetArray"`    |
| ListForm          | `"ListArray"`          |
| RecordForm        | `"RecordArray"`        |
| IndexedForm       | `"IndexedArray"`       |
| IndexedOptionForm | `"IndexedOptionArray"` |
| ByteMaskedForm    | `"ByteMaskedArray"`    |
| BitMaskedForm     | `"BitMaskedArray"`     |
| UnmaskedForm      | `"UnmaskedArray"`      |
| UnionForm         | `"UnionArray"`         |

`from_dict()` at `awkward/forms/form.py` dispatches on the `"class"` key to
reconstruct the appropriate Form subclass.

### Buffer Expectations

`_expected_from_buffers()` yields `(key, dtype)` pairs for each buffer a Form
requires:

| Form              | Buffers           |
| ----------------- | ----------------- |
| NumpyForm         | `data`            |
| EmptyForm         | (none)            |
| RegularForm       | (none)            |
| ListOffsetForm    | `offsets`         |
| ListForm          | `starts`, `stops` |
| RecordForm        | (none)            |
| IndexedForm       | `index`           |
| IndexedOptionForm | `index`           |
| ByteMaskedForm    | `mask`            |
| BitMaskedForm     | `mask`            |
| UnmaskedForm      | (none)            |
| UnionForm         | `tags`, `index`   |

Buffer keys are generated via `getkey(form, attribute_name)` which typically
produces `"{form_key}-{attribute}"` (e.g., `"node0-data"`,
`"node1-offsets"`).

## Strategy Design Implications

### Recursive Generation

Forms compose recursively. A strategy should generate form trees with controlled
depth:

- **Leaf strategies:** `NumpyForm` (primary leaf), `EmptyForm` (rare)
- **Container strategies:** wrap a recursive child form
- **Multi-child strategies:** `RecordForm` (0+ children), `UnionForm` (1+
  children)

### Index Type Selection

All index/offset/mask parameters accept any of the 5 index type strings. For
strategy generation:

- **Conventional defaults** (`i64` for offsets/index, `i8` for tags/mask) cover
  most real-world usage
- **Full exploration** can draw from all 5 types to find edge cases
- A `index_types()` strategy returning
  `st.sampled_from(['i8', 'u8', 'i32', 'u32', 'i64'])` would be useful

### Form Groupings for Strategy Design

Forms can be grouped by their role:

| Group   | Forms                                                          |
| ------- | -------------------------------------------------------------- |
| Leaf    | NumpyForm, EmptyForm                                           |
| List    | ListOffsetForm, ListForm, RegularForm                          |
| Option  | IndexedOptionForm, ByteMaskedForm, BitMaskedForm, UnmaskedForm |
| Indexed | IndexedForm                                                    |
| Record  | RecordForm                                                     |
| Union   | UnionForm                                                      |

### `simplified()` Constructors

Several forms have `simplified()` classmethods that may return a different Form
type than requested:

- `IndexedForm.simplified()` can return `IndexedOptionForm` (when content is
  option)
- `IndexedOptionForm.simplified()` can flatten nested option/indexed
- `UnmaskedForm.simplified()` can absorb indexed/option content
- `UnionForm.simplified()` delegates to Content-level simplification

**Strategy implication:** when testing `simplified()`, the output Form type may
differ from the input. Direct constructors (`__init__`) always return the
requested type.

### form_key Management

- Skip `form_key` in individual form construction (`None` is safe)
- Apply `form.form_with_unique_keys()` as post-processing on the complete form
  tree
- This matches the recommended real-world pattern

### Parameters

- Skip parameters for most forms (`None` default)
- Special cases worth testing: string/bytestring patterns (require coordinated
  `__array__` parameters on parent list form and child NumpyForm)
- `EmptyForm` cannot accept parameters at all

### Form Equality

`Form.is_equal_to()` supports three modes:

- Default: compares structure and non-`__`-prefixed parameters
- `all_parameters=True`: compares all parameters
- `form_key=True`: also compares form_key values

`UnionForm` equality uses permutation matching on contents (order-independent).

## Sources

- [Awkward Array source code](https://github.com/scikit-hep/awkward) v2.8.11
- `awkward/forms/*.py` -- all 12 Form subclass implementations
- `awkward/forms/form.py` -- base Form class, `from_dict()`, `from_type()`
- `awkward/_meta/*.py` -- Meta mixin classes
- `awkward/operations/ak_from_buffers.py` -- buffer reconstitution logic
