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
[`mike`](https://github.com/squidfunk/mike), pinned in the `docs` dep group):

```bash
mike serve --branch gh-pages
```

## PR Title Convention

This project uses [Conventional Commits](https://www.conventionalcommits.org/)
for **PR titles**. Since we squash-merge, the PR title becomes the final commit
message.

### Format

```text
type: description
```

### Allowed Types

| Type       | Purpose                                               |
| ---------- | ----------------------------------------------------- |
| `feat`     | A new feature                                         |
| `fix`      | A bug fix                                             |
| `docs`     | Documentation only                                    |
| `style`    | Code style (formatting, semicolons, etc.)             |
| `refactor` | Code change that neither fixes a bug nor adds feature |
| `perf`     | Performance improvement                               |
| `test`     | Adding or updating tests                              |
| `build`    | Build system or external dependencies                 |
| `ci`       | CI configuration                                      |
| `chore`    | Other changes that don't modify src or test files     |
| `revert`   | Reverts a previous commit                             |

### Examples

- `feat: add user authentication`
- `fix: handle empty input`
- `docs: update installation instructions`

### Individual Commits

Individual commit messages within a PR are free-form. Only the PR title is
enforced.

## For Maintainers

### Direct Commits to Main

Use the `meta:` type for commits pushed directly to main that should not appear
in the changelog or release notes, such as changes to:

- `.claude/` (Claude Code configuration)
- `.design/` (design documents)

### Releasing

Releases use a two-tag flow. The `u` tag triggers changelog generation, which in
turn creates the `v` tag and GitHub Release.

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
