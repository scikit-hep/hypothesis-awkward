# API Design: Public Namespaces of `hypothesis_awkward.strategies`

- **Date:** 2026-11-08
- **Status:** Implemented
- **Author:** Claude (with developer collaboration)
- **Related:**
  [constructors-directory-structure](../impl/2026-11-08-01-Constructors-directory-structure.md)

## Overview

This document records which strategies are public before v1.0.0 and where they
live. The top level of `hypothesis_awkward.strategies` exports only namespaces;
every public strategy is in one of three of them. Everything else is private,
reachable only by its module path.

Import convention (unchanged):

```python
import hypothesis_awkward.strategies as st_ak
```

## Before

After PR #240, `st_ak` exported 14 strategies directly and two namespaces:

- Directly: `builtin_safe_dtype_names()`, `builtin_safe_dtypes()`,
  `from_list()`, `items_from_dtype()`, `lists()`, `none_or()`, `ranges()`,
  `from_numpy()`, `numpy_arrays()`, `numpy_dtypes()`, `supported_dtype_names()`,
  `supported_dtypes()`, `numpy_types()`, and `numpy_forms()`.
- `st_ak.constructors`: `arrays()`.
- `st_ak.contents`: `contents()` and 19 other content strategies, including
  `content_lists()` and `masked_contents()`.

The subpackages were `builtins_`, `constructors`, `contents`, `forms`, `misc`,
`numpy`, and `types`.

## Decision

`st_ak` exports only the namespaces `constructors`, `numpy`, and `misc`:

| Namespace            | Public strategies                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `st_ak.constructors` | `arrays()`, `contents()`, `empty_array_contents()`, `numpy_array_contents()`, `regular_array_contents()`, `list_offset_array_contents()`, `list_array_contents()`, `record_array_contents()`, `indexed_array_contents()`, `indexed_option_array_contents()`, `byte_masked_array_contents()`, `bit_masked_array_contents()`, `unmasked_array_contents()`, `union_array_contents()`, `string_contents()`, `bytestring_contents()`, `categorical_contents()`, `leaf_contents()`, `option_contents()` |
| `st_ak.numpy`        | `supported_dtypes()`, `numpy_dtypes()`, `numpy_arrays()`                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `st_ak.misc`         | `none_or()`, `ranges()`                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |

A strategy is public if and only if it is in its namespace's `__all__`; the API
reference documents exactly these 24 strategies, one page per namespace.

### Migration

| Before                                                         | After                         |
| -------------------------------------------------------------- | ----------------------------- |
| `st_ak.contents.<name>()`                                      | `st_ak.constructors.<name>()` |
| `st_ak.supported_dtypes()`, `numpy_dtypes()`, `numpy_arrays()` | `st_ak.numpy.<name>()`        |
| `st_ak.none_or()`, `st_ak.ranges()`                            | `st_ak.misc.<name>()`         |

### Made private

Removed from `__all__` and from the API reference, still importable by module
path:

| Strategy                                  | Reason                                                                                                                                                                               |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `numpy_forms()`, `numpy_types()`          | Not ready for the public API; the form and type strategies are planned, and these are their first steps.                                                                             |
| the five `builtin` strategies             | Used only by each other and their own tests; nothing outside `builtin/` depends on them.                                                                                             |
| `from_numpy()`, `supported_dtype_names()` | Used only in tests. `numpy_arrays()` covers the public need; the dtype-name strategy samples the names of the same dtypes as `supported_dtypes()`.                                   |
| `content_lists()`, `masked_contents()`    | Internal building blocks of `contents()`, `record_array_contents()`, `union_array_contents()`, and `option_contents()`; `option_contents()` is the public strategy for option types. |
| the 12 `*_from_contents()` strategies     | Internal: `contents()` calls them to wrap children it draws. They were not in `__all__` before this change either.                                                                   |

## Rationale

### Reserve the top level

`st_ak.arrays()` is reserved for strategies that generate arrays of a given type
or form (see the roadmap). The same argument applies to `st_ak.contents()`: its
natural future meaning is "contents of a given type or form", which the current
`contents()`, structure-random and controlled by `allow_*` flags, would already
occupy.

The per-class strategies are not obviously safe at the top level either. With a
form, the class at each node is fixed, so a generator needs per-class building
blocks that take a form, a different signature from today's. With a type, the
class is not fixed (`var * int64` can be a `ListOffsetArray` or a `ListArray`),
so generation goes through a `contents(type=...)`-like entry point instead.

Moving a name up to the top level later is additive; moving it down breaks
users. While the type- and form-directed API is undesigned, the top level stays
empty of strategies.

The maturity of the content strategies was considered and rejected as a reason
for the top level: it speaks to the implementation, not to whether the names fit
the future API.

### Why `constructors` holds the content strategies

`arrays()` and every content strategy follow the same approach: they build
layouts with Awkward's direct constructors (the `arrays()` docstring cites the
Awkward Array User Guide section "Direct constructors"). One namespace for the
current approach keeps both the top level and the name `content` free for the
future counterpart of `form` and `type_`. It also matches the guides, which
already call `st_ak.constructors.arrays()`.

### Package names

- Singular `form`, `type_`, and `builtin`, replacing `forms`, `types`, and
  `builtins_`.
- `types` shadows the standard-library module `types` on a bare import, and
  `type` shadows the built-in, so the trailing underscore follows PEP 8.
- "Data type" was rejected: it means `dtype` in NumPy.
- Renaming `contents()` to `layouts()` was rejected: "layout" is Awkward's term
  for a `Content` tree, but forms and types have no equivalent word.

## Alternatives Considered

### Content strategies at the top level

Tried on the branch: the strategies for `Content` and its subclasses, plus
`string_contents()`, `bytestring_contents()`, and `categorical_contents()`, at
the top level, with `leaf_contents()` and `option_contents()` in `st_ak.content`
(later also at the top level). Rejected for the reasons in "Reserve the top
level".

### Keep the content strategies in `st_ak.content`

Rejected: `content` would hold the current approach while also being the natural
name for the counterpart of `form` and `type_`.

### Strategies exported directly from `st_ak` (status quo)

Rejected: it gave the older, less mature strategies (`lists()`, `from_numpy()`)
the shortest names, and blocked reserving names for the future API.

## Consequences

- Submodules become attributes of their namespace once imported, for example
  `st_ak.constructors.option` and `st_ak.constructors.string`. They are not in
  `__all__` and not in the API reference; `__all__` and the reference define
  what is public.
- `tests/test_import.py` checks, in a fresh interpreter, that each public
  namespace is an attribute of its parent after importing only the parent.
- Downstream code needs the migration above when it updates its pinned version.

## Open Items

- Underscore the names that are not public, so the boundary is visible in the
  names too: the 12 `*_from_contents()` strategies first, then the other
  strategies in "Made private".
- The fate of `st_ak.constructors.arrays()` once `st_ak.arrays()` lands (kept,
  wrapped, or deprecated) is undecided.
