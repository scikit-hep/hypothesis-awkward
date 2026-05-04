# Contributing

## Development Setup

Clone the repository:

```bash
git clone https://github.com/scikit-hep/hypothesis-awkward.git
cd hypothesis-awkward
```

Create a virtual environment (Python ≥ 3.10):

```bash
uv venv
source .venv/bin/activate
```

Install the project in development mode:

```bash
uv pip install -e . --group dev --group docs
```

Set up the pre-commit hooks (optional — CI enforces these on PRs):

```bash
pre-commit install
```

## Previewing Docs Locally

For a live preview of the current working tree (single version, no version
selector):

```bash
zensical serve
```

For the multi-version site as published to `gh-pages` (uses
[`mike`](https://github.com/squidfunk/mike), pinned in the `docs` dependency
group):

```bash
mike serve --branch gh-pages
```

## Docstring Style

Docstrings follow the [NumPy style] and are rendered through [mkdocstrings]
(Markdown via [griffe]).

- Inline literals — parameter names, values like `True`/`False`/`None`, dtypes,
  code snippets — use **single backticks**: `` `param_name` ``, `` `True` ``,
  `` `np.float64` ``.
- Cross-references to API objects use Markdown autorefs with single backticks
  around the visible name: ``[`NumpyArray`][ak.contents.NumpyArray]``.
- Avoid double backticks. They are an RST convention and render the same as
  single backticks in this Markdown pipeline.

[NumPy style]: https://numpydoc.readthedocs.io/en/latest/format.html
[mkdocstrings]: https://mkdocstrings.github.io/
[griffe]: https://mkdocstrings.github.io/griffe/

## PR Title Convention

This project uses [Conventional Commits](https://www.conventionalcommits.org/)
for **PR titles**. Since we squash-merge, the PR title becomes the final commit
message.

### Format

```text
type: description
```

### Allowed Types

| Type       | Purpose                                                      |
| ---------- | ------------------------------------------------------------ |
| `feat`     | New user-visible capability (new strategy, new public kwarg) |
| `fix`      | User-visible bug in production behavior                      |
| `docs`     | Documentation only                                           |
| `style`    | Code style (formatting, semicolons, etc.)                    |
| `refactor` | Code change that neither fixes a bug nor adds feature        |
| `perf`     | Performance improvement                                      |
| `test`     | Test-side change, even when fixing a flaky or incorrect test |
| `build`    | Build system or external dependencies                        |
| `ci`       | CI configuration                                             |
| `chore`    | Other changes that don't modify src or test files            |
| `revert`   | Reverts a previous commit                                    |

Append `!` to the type (e.g. `feat!:` or `fix!:`) to flag a breaking change.
Don't pair `!` with behavior-preserving types like `refactor`, `docs`, `style`,
or `test` — by definition they don't introduce breaking changes.

`fix:` and `feat:` are reserved for changes to production code's observable
behavior. If a test was failing but production behavior didn't change, use
`test:` instead (or `docs:` / `refactor:` as appropriate).

### Audience

PR titles render verbatim into `CHANGELOG.md` and release notes. The reader is a
library user skimming what changed, not a contributor who has read the diff.
Start with the user-facing name — the public strategy or kwarg the change
affects — rather than internal helpers, predicates, or test closures. The PR
body remains technical (file:line, internal names, the rationale); only the
title needs to consider its audience.

For example, ``fix: handle `max_depth=0` in `_expect_raised()` `` mentions a
private test closure no library user would recognize.
``test: cover `contents()` raise when `max_depth=0` and `min_length>0` ``
describes the same change in terms of `contents()` and the kwargs a release-note
reader scanning the changelog would look for.

### Title Style

- **Backticks for code-like terms.** Wrap module paths, file paths,
  function/class names, and other identifiers in backticks. The project's
  release tooling (`git-cliff`) preserves PR titles verbatim into
  `CHANGELOG.md`, where backticks render as inline code.
- **≤ 70 characters.** Long titles are truncated in GitHub's PR list view,
  notifications, and the changelog. If the title needs "and" to cover the
  changes, this typically indicates the PR should be split.

### Examples

- ``feat: add `min_length` to `regular_array_contents` ``
- `fix: restrict changelog link parser to commit subject line`
- `docs: cross-reference Awkward types in docstrings`
- ``feat!: drop `dicts_for_dataframe()` ``
- ``test: cover `contents()` raise when `max_depth=0` and `min_length>0` ``

### Individual Commits

Individual commit messages within a PR have no required format. Only the PR
title is enforced.

## For Maintainers

### Direct Commits to Main

Use the `meta:` type for commits pushed directly to main that should not appear
in the changelog or release notes, such as changes to:

- `.claude/` (Claude Code configuration)
- `.design/` (design documents)

### Releasing

Releases use a two-tag flow. The `u` tag triggers changelog generation, which
then creates the `v` tag and GitHub Release.

#### Steps

1. **Bump the version:**

   ```bash
   hatch version <rule>
   ```

   Where `<rule>` is `patch`, `minor`, or `major`. This updates
   `src/hypothesis_awkward/__about__.py`, creates a commit, and tags it
   `u<version>`.

2. **Push the commit and tag:**

   ```bash
   git push --follow-tags
   ```

3. **Wait for CI:**
   - The **Changelog** workflow generates `CHANGELOG.md`, commits it to `main`,
     and creates the `v` tag.
   - The **Release** workflow then creates a GitHub Release with categorized
     notes.
   - The **PyPI** workflow builds and publishes the package to PyPI.
