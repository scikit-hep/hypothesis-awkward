# Changelog

All notable changes to this project will be documented in this file.


## [0.8.1] - 2026-03-24

### Documentation

- Improve styling, footer, and page content ([#26](https://github.com/scikit-hep/hypothesis-awkward/pull/26))

### Build & CI

- Bump actions/download-artifact from 4 to 8 ([#25](https://github.com/scikit-hep/hypothesis-awkward/pull/25))
- Bump actions/github-script from 7 to 8 ([#24](https://github.com/scikit-hep/hypothesis-awkward/pull/24))
- Bump actions/upload-artifact from 4 to 7 ([#23](https://github.com/scikit-hep/hypothesis-awkward/pull/23))


## [0.8.0] - 2026-03-23

### Documentation

- Add zensical docs setup ([#20](https://github.com/scikit-hep/hypothesis-awkward/pull/20))
- Improve docs pages, module docstrings, and README ([#22](https://github.com/scikit-hep/hypothesis-awkward/pull/22))

### Build & CI

- Use reusable actions in CI workflows ([#18](https://github.com/scikit-hep/hypothesis-awkward/pull/18))
- Pin hypothesis in CI deps and drop dev group version bounds ([#19](https://github.com/scikit-hep/hypothesis-awkward/pull/19))
- Add docs build and deploy workflows ([#21](https://github.com/scikit-hep/hypothesis-awkward/pull/21))


## [0.7.5] - 2026-03-04

### Bug Fixes

- Restrict changelog link parser to commit subject line ([#17](https://github.com/scikit-hep/hypothesis-awkward/pull/17))

## [0.7.4] - 2026-03-04

### Bug Fixes

- Check out v-tag in release and PyPI workflows ([#16](https://github.com/scikit-hep/hypothesis-awkward/pull/16))

## [0.7.3] - 2026-03-04

### Bug Fixes

- Restore --prepend arg and add v0.7.2 changelog ([#15](https://github.com/scikit-hep/hypothesis-awkward/pull/15))

## [0.7.2] - 2026-03-04

### Bug Fixes

- Add missing blank line between versions in changelog ([#13](https://github.com/scikit-hep/hypothesis-awkward/pull/13))

### Refactoring

- Simplify changelog generation config ([#14](https://github.com/scikit-hep/hypothesis-awkward/pull/14))

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
