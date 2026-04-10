# Changelog

All notable changes to this project will be documented in this file.


## [0.14.0] - 2026-04-10

### Features

- Generate unreachable data in RegularArray ([#59](https://github.com/scikit-hep/hypothesis-awkward/pull/59))
- Improve RegularArray shrink quality ([#61](https://github.com/scikit-hep/hypothesis-awkward/pull/61))
- Unreachable data in RegularArray and ListOffsetArray ([#62](https://github.com/scikit-hep/hypothesis-awkward/pull/62))
- Generalize ListArray starts/stops generation ([#63](https://github.com/scikit-hep/hypothesis-awkward/pull/63))

### Documentation

- Add NumPy and Hypothesis inventories for cross-references ([#60](https://github.com/scikit-hep/hypothesis-awkward/pull/60))
- Add Guide tab and restructure into tabbed navigation ([#67](https://github.com/scikit-hep/hypothesis-awkward/pull/67))

### Refactoring

- Clean up from_contents and docstrings ([#65](https://github.com/scikit-hep/hypothesis-awkward/pull/65))


## [0.13.0] - 2026-04-06

### Features

- Improve shrink quality for leaf contents, numpy arrays, and dtypes ([#52](https://github.com/scikit-hep/hypothesis-awkward/pull/52))
- Improve regular_array defaults and docstrings ([#56](https://github.com/scikit-hep/hypothesis-awkward/pull/56))

### Bug Fixes

- Increase max_examples for rare find() condition in test ([#58](https://github.com/scikit-hep/hypothesis-awkward/pull/58))

### Testing

- Add shrink quality tests for contents() ([#57](https://github.com/scikit-hep/hypothesis-awkward/pull/57))

### Build & CI

- Switch to double-quote docstrings and add docformatter ([#53](https://github.com/scikit-hep/hypothesis-awkward/pull/53))
- Harden composite actions for gh-pages deployment ([#54](https://github.com/scikit-hep/hypothesis-awkward/pull/54))
- Split docs-pr-preview into build and deploy workflows ([#55](https://github.com/scikit-hep/hypothesis-awkward/pull/55))

### Meta

- Update CI rules, testing patterns, and option-contents design status


## [0.12.0] - 2026-04-01

### Features

- Change max_depth default from 5 to None (unbounded) ([#50](https://github.com/scikit-hep/hypothesis-awkward/pull/50))

### Bug Fixes

- Reduce max_leaf_size in test_draw_max_leaf_size to fix flaky failure ([#49](https://github.com/scikit-hep/hypothesis-awkward/pull/49))

### Documentation

- Fix docstring typos and reorder contents() parameters ([#48](https://github.com/scikit-hep/hypothesis-awkward/pull/48))


## [0.11.0] - 2026-03-31

### Features

- Rename max_size to max_leaf_size in contents() and arrays() ([#37](https://github.com/scikit-hep/hypothesis-awkward/pull/37))
- Rename content_lists() params for clarity ([#39](https://github.com/scikit-hep/hypothesis-awkward/pull/39))
- Add max_size parameter to bound total content_size() ([#40](https://github.com/scikit-hep/hypothesis-awkward/pull/40))
- Derive default max_length from size budgets ([#43](https://github.com/scikit-hep/hypothesis-awkward/pull/43))
- Add option type strategies ([#44](https://github.com/scikit-hep/hypothesis-awkward/pull/44))
- UnionArray all-or-none option coordination ([#46](https://github.com/scikit-hep/hypothesis-awkward/pull/46))

### Bug Fixes

- Increase max_examples for flaky test_draw_nan ([#38](https://github.com/scikit-hep/hypothesis-awkward/pull/38))

### Documentation

- Add util module to API reference ([#41](https://github.com/scikit-hep/hypothesis-awkward/pull/41))
- Update arrays() and contents() docstrings ([#47](https://github.com/scikit-hep/hypothesis-awkward/pull/47))

### Refactoring

- Extract case blocks from contents() into strategy modules ([#42](https://github.com/scikit-hep/hypothesis-awkward/pull/42))

### Testing

- Add section comments to property-based tests ([#35](https://github.com/scikit-hep/hypothesis-awkward/pull/35))

### Meta

- Add option types research and API design documents
- Update option type design docs for max_size budget


## [0.10.0] - 2026-03-27

### Features

- Change allow_nan default from False to True ([#34](https://github.com/scikit-hep/hypothesis-awkward/pull/34))

### Bug Fixes

- Add assert for dtype.names narrowing in structured dtype branches ([#32](https://github.com/scikit-hep/hypothesis-awkward/pull/32))

### Documentation

- Add maintainer section and document meta commit type ([#28](https://github.com/scikit-hep/hypothesis-awkward/pull/28))
- Add conda-forge badge to README ([#29](https://github.com/scikit-hep/hypothesis-awkward/pull/29))

### Build & CI

- Bump codecov/codecov-action from 5 to 6 ([#30](https://github.com/scikit-hep/hypothesis-awkward/pull/30))
- Add pre-commit with ruff hooks ([#31](https://github.com/scikit-hep/hypothesis-awkward/pull/31))
- Replace dprint with prettier for markdown formatting ([#33](https://github.com/scikit-hep/hypothesis-awkward/pull/33))

### Meta

- Add worktree rules for Claude Code
- Remove personal rules from shared .claude/


## [0.9.0] - 2026-03-25

### Features

- Update development status from Alpha to Beta ([#27](https://github.com/scikit-hep/hypothesis-awkward/pull/27))


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
