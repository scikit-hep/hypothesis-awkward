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
then creates the `v` tag, the GitHub Release, and the versioned docs deployment.

#### Steps

1. **Check out the commit to release:**

   ```bash
   git switch main && git pull
   ```

   Or, to release from an older commit, `git switch --detach <commit>`. The
   chosen commit must already contain the release workflow files: a tag push
   runs the workflows as of the tagged commit.

2. **Bump the version:**

   ```bash
   hatch version <rule>
   ```

   Where `<rule>` is `patch`, `minor`, or `major`. This updates
   `src/hypothesis_awkward/__about__.py`, creates a commit, and tags it
   `u<version>`.

3. **Push only the tag:**

   ```bash
   git push origin u<version>
   ```

   Not `main --tags`: a stale local `latest` tag makes `--tags` fail. GitHub
   Actions pushes the changelog commit and the other tags back to you.

4. **Wait for CI:**

   - The **Generate changelog** workflow creates a `release/<version>` branch at
     the tagged commit, generates `CHANGELOG.md` and the `v` tag on it, then
     merges the branch back into `main` and deletes it (a fast-forward when
     released from `main`'s head, a merge commit otherwise). A backport — a
     release cut from a commit off `main`'s line — skips the merge-back with a
     warning and keeps the branch as the maintenance line; `main` is untouched.
   - The **Release a new version** workflow creates a GitHub Release with
     categorized notes, marked as the latest release only when the version is
     the newest by version order.
   - The **Upload to PyPI** workflow builds and publishes the package to PyPI.
   - The **Deploy release docs** workflow publishes the versioned docs; the
     `latest` alias and the default version move only when the version is the
     newest.

5. **Pull the results:** after a merge-back,
   `git pull --tags --force origin main` (`--force` lets the moved `latest` tag
   update). After a backport, `main` has nothing new;
   `git fetch --tags --force origin` retrieves the branch and the tags.

#### Backports

A release whose merge-back was skipped keeps its `release/<version>` branch as
the maintenance line. To cut the next release on that line, check out its
`v`-tagged commit (`git switch --detach v<version>`) and follow the same steps.
`main`'s `CHANGELOG.md` never lists backports, and the docs `latest` alias stays
on the newest release. Run one release at a time.

#### If the release fails

A failed **Generate changelog** run leaves a skipped **Release a new version**
run and no release. Delete the trigger tag
(`git push origin --delete u<version>`), plus the `release/<version>` branch and
the `v<version>` tag if the failed run created them. Fix the cause and push the
trigger tag again. A pre-existing `release/<version>` ref makes the branch
creation fail loudly by design.
