---
paths:
  - ".github/**"
  - "pyproject.toml"
---

# CI and Dependency Management

## Dependency Pinning

The project uses two requirements files under `.github/deps/` for CI testing:

- **`minimum/requirements.txt`** — Pins the oldest supported versions (matching
  the lower bounds in `pyproject.toml`). Manually maintained.
- **`latest/requirements.txt`** — Pins the latest known versions. Monitored by
  Dependabot, which opens PRs to bump these pins.

These files are **not** used for local development. The project does not use
`uv.lock`.

## Updating Dependencies

- When bumping a lower bound in `pyproject.toml` (e.g., `awkward>=2.9`), also
  update `minimum/requirements.txt` to match.
- `latest/requirements.txt` is updated automatically by Dependabot.

## CI Matrix (`unit-test.yml`)

Runs on push to `main` and PRs targeting `main`. Tests a matrix of:

- **Python**: 3.10, 3.11, 3.12, 3.13, 3.14
- **Deps**: `default`, `latest`, `min`
- **Optional deps**: none or `[all]` (pandas + pyarrow)

The `[all]` extra only runs with `default` deps (excluded from `latest`/`min`).

| Mode      | Install method                                                    |
| --------- | ----------------------------------------------------------------- |
| `default` | `uv pip install -e .[opt-deps] --group dev` (free resolution)     |
| `latest`  | Installs package, then overwrites with `latest/requirements.txt`  |
| `min`     | Installs package, then downgrades with `minimum/requirements.txt` |

## Release Workflows (Two-Tag Pipeline)

Releases use a `u`/`v` two-tag flow to work around the `GITHUB_TOKEN` limitation
that prevents one workflow from triggering another via tag push:

1. **`hatch version <rule>`** — bumps the version, commits, and creates a
   `u<version>` tag (configured via `tag_name` in `pyproject.toml`).
2. **`changelog.yml`** — triggered by the `u*.*.*` tag push. Generates
   `CHANGELOG.md` with git-cliff, commits the result, creates the corresponding
   `v<version>` tag, and pushes both to `main`.
3. **`release.yml`** — triggered by `workflow_run` on the changelog workflow.
   Creates a GitHub Release with auto-generated notes and updates a `latest`
   tag.
4. **`pypi.yml`** — triggered by `workflow_run` on the changelog workflow.
   Builds with `hatch build` and publishes to PyPI via trusted publishing.

### Documentation Workflows

Release and dev deploys use the Zensical-compatible fork of
[`mike`](https://github.com/squidfunk/mike), pinned in the `docs` dependency
group to tag `2.2.0+zensical-0.1.0`. `mike` writes per-version subdirectories
plus a `versions.json` manifest and a root redirect on the `gh-pages` branch. PR
previews are deployed outside `mike`'s manifest under `pr/<N>/` and are
invisible to the version selector.

- **`docs-dev.yml`** — Runs `mike deploy --push dev` on push to `main`,
  refreshing the `dev/` subdirectory and its `versions.json` entry.
- **`docs-release.yml`** — Runs
  `mike deploy --push --update-aliases <version> latest` then
  `mike set-default --push latest` on the `workflow_run` after changelog
  generation.
- **`docs-pr-build.yml`** — Builds a single-version preview with the mike
  version provider stripped via
  `.github/actions/build-docs/prepare_pr_build.py`. Uploads the built site as an
  artifact (no write access to `gh-pages`).
- **`docs-pr-deploy.yml`** — Downloads the artifact and writes it to `pr/<N>/`
  on `gh-pages`, then comments the preview URL on the PR.
- **`docs-pr-cleanup.yml`** — Removes `pr/<N>/` when the PR closes.

#### Bootstrap

`.github/scripts/bootstrap-mike-versions.sh` is a one-time script that seeds
`versions.json` from the existing numeric subdirectories on `gh-pages`, rewrites
`latest/` as a symlink to the newest version, and runs `mike set-default latest`
to create the root redirect. Not wired to any workflow; run manually once per
repository.

#### Migration off the fork

The `squidfunk/mike` fork is explicitly temporary. When Zensical ships native
versioning, remove the `mike` dep, replace the workflow commands with the native
equivalents, and delete the bootstrap script. Track
<https://zensical.org/about/roadmap/#versioning>.

### PR Workflows

- **`pr-title.yml`** — Validates PR titles follow Conventional Commits format.
- **`conventional-label.yml`** — Auto-labels PRs based on the commit type in the
  title (e.g., `feat:` -> `feature` label).
- **`pre-commit.yml`** — Runs pre-commit hooks (ruff, etc.) on PRs.
