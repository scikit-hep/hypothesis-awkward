# Directory Structure: `types` Module

**Date:** 2026-01-21
**Status:** Draft
**Related:** [types-api.md](./../api/2026-01-21-types-api.md)

## Overview

This document proposes the directory structure for implementing the `types()`
strategy and its supporting strategies.

## Proposed Structure

### Source (`src/hypothesis_awkward/strategies/types/`)

```text
src/hypothesis_awkward/strategies/types/
├── __init__.py      # Re-exports all public strategies
├── types.py         # Main types() strategy (recursive composition)
├── numpy_.py        # numpy_types()
├── list_.py         # list_types()
├── regular.py       # regular_types()
├── option.py        # option_types()
├── record.py        # record_types()
├── union.py         # union_types()
└── string.py        # string_types(), bytestring_types()
```

### Tests (`tests/strategies/types/`)

```text
tests/strategies/types/
├── __init__.py
├── test_types.py         # Main types() strategy tests
├── test_numpy_types.py   # numpy_types() tests
├── test_list_types.py    # list_types() tests
├── test_regular_types.py # regular_types() tests
├── test_option_types.py  # option_types() tests
├── test_record_types.py  # record_types() tests
├── test_union_types.py   # union_types() tests
└── test_string_types.py  # string_types(), bytestring_types() tests
```

## Design Rationale

### One File Per Type Class

**Decision:** Each Awkward type class gets its own module file.

**Rationale:**

- **Matches existing patterns:** `strategies/numpy/` has `numpy.py` and
  `dtype.py`; `strategies/builtins_/` has `list_.py` and `dtype.py`
- **Enables incremental TDD:** Implement and test one type at a time
- **Clear ownership:** Easy to find where a specific type strategy lives
- **Manageable file sizes:** Each file stays focused and readable

### Trailing Underscore Convention

**Decision:** Use `numpy_.py` and `list_.py` (with trailing underscore).

**Rationale:**

- Avoids shadowing built-in `list` and standard library `numpy`
- Consistent with existing `strategies/builtins_/` naming
- Python convention for avoiding keyword/builtin conflicts

### Separate `string.py` for String Types

**Decision:** Group `string_types()` and `bytestring_types()` in one file.

**Rationale:**

- Both are semantically related (text-like types)
- Both are implemented as `ListType` with special parameters
- Small enough to share a file without confusion

### Test File Naming

**Decision:** Test files use `test_<strategy_name>.py` pattern.

**Rationale:**

- Matches existing pattern: `test_numpy_arrays.py`, `test_ranges.py`
- Clear 1:1 mapping between source and test files
- Easy to run specific tests: `pytest tests/strategies/types/test_numpy_types.py`

## Alternatives Considered

### Alternative A: Single File

```text
src/hypothesis_awkward/strategies/types.py  # Everything in one file
```

**Rejected because:**

- Would become too large (8+ strategies)
- Harder to navigate and maintain
- Doesn't match existing multi-file module pattern

### Alternative B: Grouped by Complexity

```text
src/hypothesis_awkward/strategies/types/
├── __init__.py
├── leaf.py        # numpy_types(), string_types(), bytestring_types()
├── container.py   # list_types(), regular_types(), option_types()
├── compound.py    # record_types(), union_types()
└── types.py       # Main types() strategy
```

**Rejected because:**

- Groupings are somewhat arbitrary
- Harder to find specific strategies
- One-file-per-type is more intuitive

### Alternative C: Flat Structure (No Subdirectory)

```text
src/hypothesis_awkward/strategies/
├── types.py           # Main types() strategy
├── numpy_types.py     # numpy_types()
├── list_types.py      # list_types()
...
```

**Rejected because:**

- Clutters the `strategies/` directory
- Doesn't group related functionality
- Inconsistent with `numpy/`, `builtins_/`, `pandas/` pattern

## Implementation Order

For TDD, implement in this order (simplest to most complex):

1. **`numpy_.py`** - Leaf type, no recursion, foundation for others
2. **`string.py`** - Simple, deterministic output
3. **`list_.py`** - First container type, takes content strategy
4. **`regular.py`** - Similar to list, adds size parameter
5. **`option.py`** - Simple wrapper type
6. **`record.py`** - Adds field name generation complexity
7. **`union.py`** - Multiple contents, variant constraints
8. **`types.py`** - Composes all above with recursive depth control

Each step: write tests first, then implement to pass tests.

## Public API Exports

The `__init__.py` should export:

```python
from hypothesis_awkward.strategies.types.list_ import list_types
from hypothesis_awkward.strategies.types.numpy_ import numpy_types
from hypothesis_awkward.strategies.types.option import option_types
from hypothesis_awkward.strategies.types.record import record_types
from hypothesis_awkward.strategies.types.string import bytestring_types, string_types
from hypothesis_awkward.strategies.types.types import types
from hypothesis_awkward.strategies.types.union import union_types

__all__ = [
    'bytestring_types',
    'list_types',
    'numpy_types',
    'option_types',
    'record_types',
    'string_types',
    'types',
    'union_types',
]
```

These should also be re-exported from `strategies/__init__.py` for the
`import hypothesis_awkward.strategies as st_ak` usage pattern.
