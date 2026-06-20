---
name: audit-dot-design
description: Audit `.design/` docs against the repo and report divergence
disable-model-invocation: true
---

Audit the design records in `.design/` against the current state of the
repository. These are dated docs (research, API designs, implementation notes) —
the goal is to surface and report where they have diverged from the code.
Reconciling each divergence is followed up interactively.

## Steps

1. **Explore the repo** — Use Glob and Read to understand the current source
   layout (`src/`), tests (`tests/`), and config files (`pyproject.toml`,
   `.github/`).

2. **Audit `.design/`** — Read every file under `.design/` (`research/`, `api/`,
   `impl/`, `notes/`). For each file, check whether the described API,
   implementation plan, or research is still consistent with the current code.
   Flag anything that has diverged. Use the dated naming convention in
   `.design/README.md` to read files in chronological order (a newer doc may
   supersede an older one).

3. **Report findings** — Summarize what has diverged and what notes you propose.
   Group by file or subdirectory.
