#!/usr/bin/env bash
# One-time bootstrap of mike's versions.json onto an existing gh-pages
# branch that already has per-version subdirectories but no manifest.
#
# Usage:
#   bash .github/scripts/bootstrap-mike-versions.sh [BRANCH]
#
# BRANCH defaults to "gh-pages". The script:
#   1. Forces the local BRANCH to origin/BRANCH and checks it out in a
#      temp worktree.
#   2. Writes versions.json listing all numeric version dirs on the
#      branch (newest first), plus a "dev" entry if the dir exists; the
#      newest numeric version gets the "latest" alias.
#   3. Replaces the existing "latest/" tree with a git symlink to the
#      newest numeric version (matches mike's alias_type=symlink output).
#   4. Commits the manifest, then runs `mike set-default --push latest`
#      to rewrite the root index.html and push both commits upstream.
#
# After the push, verify with:
#   uv run --group docs mike list --branch gh-pages
#
# Requires:
#   - `uv` with the `docs` dep group installed
#   - push access to origin
set -euo pipefail

# The gh-pages branch has no .pre-commit-config.yaml; allow commits anyway.
export PRE_COMMIT_ALLOW_NO_CONFIG=1

BRANCH="${1:-gh-pages}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKTREE="$(mktemp -d)/${BRANCH//\//-}-bootstrap"

# Resolve mike's absolute path via the project venv so we can invoke it
# from any cwd. `uv run` requires the project dir as cwd.
MIKE="$(cd "$PROJECT_DIR" && uv run --group docs -- which mike)"

cleanup() {
    git -C "$PROJECT_DIR" worktree remove --force "$WORKTREE" >/dev/null 2>&1 || true
}
trap cleanup EXIT

cd "$PROJECT_DIR"
git fetch origin "$BRANCH"
# -B forces a local branch to origin/$BRANCH so commits are reachable
# via the branch ref, not just a detached HEAD.
git worktree add -B "$BRANCH" "$WORKTREE" "origin/$BRANCH"

cd "$WORKTREE"

python3 - <<'PY'
import json
import re
from pathlib import Path

from packaging.version import InvalidVersion, Version

entries = []
numeric: list[tuple[Version, str]] = []
for d in Path('.').iterdir():
    if not d.is_dir() or not re.match(r'\d', d.name):
        continue
    try:
        numeric.append((Version(d.name), d.name))
    except InvalidVersion:
        continue
numeric.sort(key=lambda item: item[0], reverse=True)

if Path('dev').is_dir():
    entries.append({
        'version': 'dev',
        'title': 'dev',
        'aliases': [],
        'properties': {},
    })

for i, (_, name) in enumerate(numeric):
    entries.append({
        'version': name,
        'title': name,
        'aliases': ['latest'] if i == 0 else [],
        'properties': {},
    })

Path('versions.json').write_text(json.dumps(entries, indent=2) + '\n')
PY

NEWEST="$(python3 -c '
import json, pathlib
for e in json.loads(pathlib.Path("versions.json").read_text()):
    if "latest" in e["aliases"]:
        print(e["version"])
        break
')"

if [[ -z "$NEWEST" ]]; then
    echo "No numeric version dir found; nothing to alias as latest." >&2
    exit 1
fi

if [[ -e latest || -L latest ]]; then
    git rm -rf --quiet latest
fi
ln -s "$NEWEST" latest

git add versions.json latest
if git diff --cached --quiet; then
    echo "versions.json and latest already up to date; skipping bootstrap commit."
else
    git commit -m "meta: bootstrap mike versions.json"
fi

# Run mike from the project dir so it can read zensical.toml. mike
# writes to the branch via git plumbing, independent of cwd.
cd "$PROJECT_DIR"
"$MIKE" set-default --push --branch "$BRANCH" latest

echo
echo "Bootstrap complete. Verify with:"
echo "  uv run --group docs mike list --branch $BRANCH"
