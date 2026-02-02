# Directory Structure: `forms` Module

**Date:** 2026-02-02
**Status:** Draft
**Related:** [forms-api.md](./../api/2026-02-02-forms-api.md)

## Overview

This document proposes the directory structure for implementing the
`numpy_forms()` strategy, designed for future extension to a full `forms()`
strategy module.

## Proposed Structure

### Source (`src/hypothesis_awkward/strategies/forms/`)

```text
src/hypothesis_awkward/strategies/forms/
├── __init__.py      # Re-exports all public strategies
└── numpy_.py        # numpy_forms()
```

Future files (not implemented now):

```text
src/hypothesis_awkward/strategies/forms/
├── __init__.py      # Re-exports all public strategies
├── forms.py         # Main forms() strategy (recursive composition)
├── numpy_.py        # numpy_forms()
├── list_.py         # list_forms() — ListOffsetForm, ListForm
├── regular.py       # regular_forms()
├── option.py        # option_forms() — IndexedOptionForm, ByteMaskedForm, etc.
├── record.py        # record_forms()
├── union.py         # union_forms()
└── string.py        # string_forms(), bytestring_forms()
```

### Tests (`tests/strategies/forms/`)

```text
tests/strategies/forms/
├── __init__.py
└── test_numpy_forms.py   # numpy_forms() tests
```

Future files (not implemented now):

```text
tests/strategies/forms/
├── __init__.py
├── test_forms.py         # Main forms() strategy tests
├── test_numpy_forms.py   # numpy_forms() tests
├── test_list_forms.py    # list_forms() tests
├── test_regular_forms.py # regular_forms() tests
├── test_option_forms.py  # option_forms() tests
├── test_record_forms.py  # record_forms() tests
├── test_union_forms.py   # union_forms() tests
└── test_string_forms.py  # string_forms(), bytestring_forms() tests
```

## Design Rationale

### Mirror the `types/` Directory Structure

**Decision:** Use the same one-file-per-form-class layout as `types/`.

**Rationale:**

- **Consistency:** `types/` uses `numpy_.py`, `list_.py`, etc.; `forms/` should
  match
- **Proven pattern:** The `types/` layout is already established and working
- **Predictability:** Developers can guess where to find a form strategy based
  on the types layout

### Trailing Underscore Convention

**Decision:** Use `numpy_.py` and `list_.py` (with trailing underscore).

**Rationale:**

- Matches `types/numpy_.py` naming
- Avoids shadowing `numpy` and `list`
- Consistent with `strategies/builtins_/` naming

### Start with `numpy_.py` Only

**Decision:** Create only `__init__.py` and `numpy_.py` initially.

**Rationale:**

- `NumpyForm` is the leaf form, required by all other forms
- Matches the `types/` implementation order (started with `numpy_.py`)
- Enables incremental TDD without placeholder files
- Future files are documented here for reference but not created

## Public API Exports

### `forms/__init__.py` (initial)

```python
__all__ = [
    'numpy_forms',
]

from .numpy_ import numpy_forms
```

### `strategies/__init__.py` Addition

Add to the existing `strategies/__init__.py`:

```python
from .forms import numpy_forms
```

And add `'numpy_forms'` to `__all__`.

This follows the existing pattern where `numpy_types` is imported from `.types`
and re-exported.

### Import Convention

Users import as:

```python
import hypothesis_awkward.strategies as st_ak

f = st_ak.numpy_forms().example()
```

## Comparison with `types/` Structure

| Aspect         | `types/`                  | `forms/`                  |
| -------------- | ------------------------- | ------------------------- |
| Module root    | `strategies/types/`       | `strategies/forms/`       |
| Leaf strategy  | `numpy_.py`               | `numpy_.py`               |
| Main strategy  | `types.py`                | `forms.py` (future)       |
| Naming         | `*_types()`               | `*_forms()`               |
| Test directory | `tests/strategies/types/` | `tests/strategies/forms/` |
| Test naming    | `test_*_types.py`         | `test_*_forms.py`         |

## Implementation Order

For TDD, implement in this order:

1. **`numpy_.py`** -- Leaf form, no recursion, foundation for others

Future (not in scope now):

2. **`string.py`** -- String/bytestring forms (ListOffsetForm with parameters)
3. **`list_.py`** -- ListOffsetForm, ListForm
4. **`regular.py`** -- RegularForm
5. **`option.py`** -- IndexedOptionForm, ByteMaskedForm, BitMaskedForm
6. **`record.py`** -- RecordForm
7. **`union.py`** -- UnionForm
8. **`forms.py`** -- Composes all above with recursive depth control

Each step: write tests first, then implement to pass tests.
