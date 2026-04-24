# CLAUDE.md

## Project Overview

Python library providing [Hypothesis](https://hypothesis.works/) strategies for
generating [Awkward Array](https://awkward-array.org/) test data.

## Build and Development Commands

For basic dev setup, see `CONTRIBUTING.md`.

### Bash tool persistence model

Each Bash tool call spawns a fresh shell.

- **Persists** across calls: working directory (`cd` carries over) and env vars
  inherited at session launch (e.g. `VIRTUAL_ENV` if Claude Code was launched
  from an activated shell).
- **Does not persist**: `export`, `source`, aliases. So
  `source .venv/bin/activate` inside a call has no effect on the next call.

### Session orientation check

Run at session start and whenever you suspect cwd has drifted (`cd` persists
across calls, so silent mispositioning is possible):

```bash
pwd
[ "$VIRTUAL_ENV" = "$PWD/.venv" ] && echo in-venv || echo use-uv-run
```

- `pwd` should be the repo root (or a subdirectory of it).
- `in-venv` → run commands bare.
- `use-uv-run` → prefix every command with `unset VIRTUAL_ENV && uv run`. The
  `unset` is a no-op when `VIRTUAL_ENV` is already empty, and prevents a stale
  inherited value (e.g. the IDE set it for the main worktree while Claude runs
  in a linked worktree) from misleading `uv run`.

### Common commands

```bash
pytest -vv --cov
pytest tests/strategies/numpy/test_from_numpy.py -vv
pytest tests/strategies/numpy/test_from_numpy.py::test_from_numpy -vv
mypy src tests
ruff check src tests
ruff format src tests
```

### Bootstrapping a new linked worktree

Only if the worktree has no `.venv` yet. Match the main worktree's Python
version:

```bash
cd ../hypothesis-awkward-<topic>
uv venv --python "$(awk '/^version_info/{print $3}' ../hypothesis-awkward/.venv/pyvenv.cfg)"
uv pip install --python .venv/bin/python -e . --group dev --group docs
```

`--python .venv/bin/python` targets the new venv explicitly, bypassing any
inherited `VIRTUAL_ENV`.

## Code Architecture

### Source Layout (`src/hypothesis_awkward/`)

- **`strategies/`** — public Hypothesis strategies (imported as
  `hypothesis_awkward.strategies`). Subpackages: `numpy/`, `builtins_/`,
  `contents/`, `constructors/`, `forms/`, `types/`, `misc/`.
- **`util/`** — internal utilities for dtype handling, safe comparisons, layout
  iteration, and array introspection.

### Strategy Design Pattern

Each strategy module pairs:

1. A "raw" strategy generating intermediate data (e.g., `numpy_arrays` → NumPy
   arrays).
2. A convenience strategy converting to Awkward (e.g., `from_numpy` wraps
   `numpy_arrays` and calls `ak.from_numpy`).

### Import Convention

```python
import hypothesis_awkward.strategies as st_ak
```

## Testing

Tests mirror `src/` under `tests/`. pytest, with doctest enabled for docstrings
and markdown files.

## Design Documents

In `.design/`; see `.design/README.md` for the index.

## Conventional Commits

PR titles follow [Conventional Commits](https://www.conventionalcommits.org/),
no scopes. Allowed types and the release procedure are in `CONTRIBUTING.md`.

## Code Style

- Docstrings: NumPy-style, double quotes (`"""`). Other strings: single quotes.
- Use absolute imports for parent packages (e.g.,
  `from hypothesis_awkward.strategies.numpy import numpy_arrays`, not
  `from ..numpy import numpy_arrays`). Same-package relative imports (e.g.,
  `from .arrays_ import arrays`) are fine.
- For forward references in type annotations, prefer string quotes
  (`-> 'MyClass[Any]'`) over `from __future__ import annotations`.
- Prefer generic constructor calls over annotated assignments for typed empty
  collections (`list[int]()` over `x: list[int] = []`).
- Place private or supportive functions after the main public ones.
- Wrap docstring text to 88 columns (docformatter enforces).
- Sort imports: `ruff check --select I --fix src tests`.
- Type-check: `mypy src tests`.
