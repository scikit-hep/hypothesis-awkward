# Changelog

All notable changes to this project will be documented in this file.

## [0.7.1] - 2026-03-03

### Documentation

- Clean up historical CHANGELOG.md entries ([#12](https://github.com/scikit-hep/hypothesis-awkward/pull/12))

### Build & CI

- Switch changelog generation to prepend mode ([#11](https://github.com/scikit-hep/hypothesis-awkward/pull/11))
- Bump pandas from 3.0.0 to 3.0.1 in /.github/deps/latest ([#8](https://github.com/scikit-hep/hypothesis-awkward/pull/8))

## [0.7.0] - 2026-03-03

### Build & CI

- Add automated changelog and release workflows ([#9](https://github.com/scikit-hep/hypothesis-awkward/pull/9))
- Add conventional commit prefixes to Dependabot PR titles ([#10](https://github.com/scikit-hep/hypothesis-awkward/pull/10))

### Pre-Conventional Changes

- Add `max_length` parameter to `arrays()` and content strategies
- Rewrite `contents()` tree builder to a top-down approach with nested union support
- Add `content_lists()` strategy for multi-child node building
- Add `max_size` and `max_zeros_length` parameters to `regular_array_contents()`
- Add `max_size` parameter to `content_lists()`

## [0.6.1] - 2026-02-19

Baseline release prior to Conventional Commits adoption. Summary of the
library at this point:

- `from_numpy()` and `from_list()` strategies for generating Awkward Arrays
  from NumPy arrays and Python lists
- `arrays()` high-level strategy with configurable content types, nesting
  depth, and virtual array support
- Content strategies for all core Awkward layouts: `numpy_array_contents()`,
  `list_offset_array_contents()`, `list_array_contents()`,
  `regular_array_contents()`, `record_array_contents()`,
  `union_array_contents()`, `string_contents()`, `bytestring_contents()`, and
  `empty_array_contents()`
- `contents()` recursive tree builder for nested array structures
- `leaf_contents()` strategy for leaf-node content selection
- `numpy_dtypes()` and `supported_dtypes()` for NumPy dtype generation
- `numpy_types()` and `numpy_forms()` for Awkward type- and form-based
  generation
- `dicts_for_dataframe()` for pandas DataFrame integration
- Layout traversal utilities: `iter_contents()`, `iter_leaf_contents()`,
  `iter_numpy_arrays()`
- Utility strategies and helpers: `none_or()`, `ranges()`, `RecordDraws`,
  `OptsChain`
