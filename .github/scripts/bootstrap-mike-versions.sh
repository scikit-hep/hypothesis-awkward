#!/usr/bin/env bash
# One-time bootstrap of mike's versions.json onto an existing gh-pages
# branch that already has per-version subdirectories but no manifest.
#
# Usage:
#   bash .github/scripts/bootstrap-mike-versions.sh [BRANCH]
#
# BRANCH defaults to "gh-pages". The script:
#   1. Creates a temporary worktree for BRANCH at /tmp/<branch>-bootstrap.
#   2. Writes versions.json listing all numeric version dirs on the branch
#      (newest first), plus a "dev" entry; the newest numeric version gets
#      the "latest" alias.
#   3. Replaces the existing "latest/" tree with a git symlink to the
#      newest numeric version (matches mike's alias_type=symlink output).
#   4. Runs `mike set-default --push latest` to rewrite the root
#      index.html as a redirect and push the result upstream.
#
# After the push, verify with:
#   uv run --group docs mike list --branch gh-pages
#
# Requires:
#   - `uv` with the `docs` dep group installed (see pyproject.toml)
#   - push access to origin
set -euo pipefail

# The gh-pages branch has no .pre-commit-config.yaml; allow commits anyway.
export PRE_COMMIT_ALLOW_NO_CONFIG=1

BRANCH="${1:-gh-pages}"
WORKTREE="$(mktemp -d)/${BRANCH//\//-}-bootstrap"

cleanup() {
    git worktree remove --force "$WORKTREE" >/dev/null 2>&1 || true
}
trap cleanup EXIT

git fetch origin "$BRANCH"
git worktree add "$WORKTREE" "origin/$BRANCH"

cd "$WORKTREE"

# Collect numeric version dirs (descending), and write versions.json.
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

# Replace any existing latest/ tree with a symlink pointing at $NEWEST.
if [[ -e latest || -L latest ]]; then
    git rm -rf --quiet latest
fi
ln -s "$NEWEST" latest

git add versions.json latest
git commit -m "meta: bootstrap mike versions.json"

# Generate the root redirect index.html via mike set-default and push.
# --branch is a local ref; we're already on the branch in this worktree.
uv run --group docs mike set-default --push --branch "$BRANCH" latest

echo
echo "Bootstrap complete. Verify with:"
echo "  uv run --group docs mike list --branch $BRANCH"
