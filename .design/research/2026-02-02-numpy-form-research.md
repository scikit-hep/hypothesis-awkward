# NumpyForm Research

**Date:** 2026-02-02
**Purpose:** Inform the design of a `numpy_forms()` strategy for hypothesis-awkward

## Constructor

```python
NumpyForm(primitive, inner_shape=(), *, parameters=None, form_key=None)
```

**Location:** `/awkward/forms/numpyform.py:49-67`

Validation:

- `primitive` — normalized via `dtype_to_primitive(primitive_to_dtype(primitive))`
  round-trip; raises `TypeError` for unrecognized primitives
- `inner_shape` — must be iterable; converted to tuple; no validation on element
  values
- `parameters` — must be `dict` or `None` (validated by `Form._init`)
- `form_key` — must be `str` or `None` (validated by `Form._init`)

## inner_shape

- **Type:** tuple of positive integers (or `unknown_length` singleton, rare)
- **Default:** empty tuple `()`
- **Non-empty when:** NumPy dtype has a subdtype (fixed-size sub-arrays)

### How it is populated

`from_dtype()` at `/awkward/forms/numpyform.py:23-44`:

```python
if dtype.subdtype is None:
    inner_shape = ()
else:
    inner_shape = dtype.shape
    dtype = dtype.subdtype[0]
```

Example: `np.dtype((np.int32, (2, 3)))` produces `inner_shape = (2, 3)`.

`NumpyArray.inner_shape` at `/awkward/contents/numpyarray.py:181-186` is
`array.shape[1:]` (all dimensions except the first).

### How it is used

`NumpyForm.type` at `/awkward/forms/numpyform.py:137-147` wraps the base
`NumpyType` in `RegularType` layers (reversed):

```python
out = ak.types.NumpyType(self._primitive, parameters=None)
for x in self._inner_shape[::-1]:
    out = ak.types.RegularType(out, x)
out._parameters = self._parameters
```

Example: `NumpyForm('int32', (2, 3)).type` produces
`RegularType(RegularType(NumpyType('int32'), 3), 2)`.

`ak.from_buffers` at `/awkward/operations/ak_from_buffers.py:305-335` multiplies
buffer size by `math.prod(form.inner_shape)` and reshapes data to
`(length, *form.inner_shape)`.

## form_key

- **Type:** `str` or `None`
- **Default:** `None`
- **Purpose:** identifier for buffer management in serialization/deserialization
- **Uniqueness:** not enforced, though convention suggests uniqueness within a
  form tree
- **Safe to omit:** yes; most tests use `None`

Used by `ak.from_buffers()` via `getkey(form, "data")` to map forms to data
buffers.

## NumpyMeta mixin

**Location:** `/awkward/_meta/numpymeta.py:12-56`

Pure metadata mixin — no validation. Contributes:

- `is_numpy = True`, `is_leaf = True`
- `purelist_parameters(*keys)` — retrieves parameters
- `purelist_isregular` — always `True`
- `purelist_depth` — `len(inner_shape) + 1`
- `minmax_depth`, `branch_depth` — cached depth properties
- `fields`, `is_tuple`, `dimension_optiontype` — leaf-related properties

## Form base class

**Location:** `/awkward/forms/form.py:384-406`

`_init` validates and stores `parameters` and `form_key`. Provides:

- `type` property (abstract, implemented by NumpyForm)
- `to_dict()`, `to_json()` for serialization
- `copy()`, `is_equal_to()` for cloning and comparison
- `columns()`, `select_columns()`, `column_types()` for structure traversal
- `length_zero_array()`, `length_one_array()` for creating test arrays

## Relationship to NumpyArray

- `NumpyArray.form_cls = NumpyForm` at `/awkward/contents/numpyarray.py:148`
- `NumpyArray._form_with_key()` at `/awkward/contents/numpyarray.py:199-213`
  creates NumpyForm from a NumpyArray
- NumpyForm can exist independently of NumpyArray — it is a lightweight
  descriptor

## Real-world usage patterns

| Pattern          | Example                                                |
| ---------------- | ------------------------------------------------------ |
| Simple primitive | `NumpyForm('int32')`                                   |
| With inner shape | `NumpyForm('int32', (2, 3))`                           |
| With parameters  | `NumpyForm('uint8', parameters={'__array__': 'char'})` |
| From dtype       | `from_dtype(np.dtype((np.float64, (3,))))`             |

Most common: simple primitives with empty inner_shape, no parameters, no
form_key.

## Strategy design implications

- **primitive:** reuse existing `supported_dtypes()` strategy
- **inner_shape:** generate empty tuple (common) or small tuples of positive
  integers; max dimensions ~1-3; dimension sizes ~2-10
- **parameters:** skip for now (`None`)
- **form_key:** skip for now (`None`)
