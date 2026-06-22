---
paths:
  - "docs/**"
---

# Documentation

Operational conventions for pages under `docs/`. Strategy and the page backlog
are in `.design/notes/2026-06-17-02-Docs-plan.md`; voice rules are in
`docs-voice.md`; build and deploy workflows are in `ci.md`.

## Authoring

To author or substantially revise a page, use the `write-docs-page` skill (the
persona-review workflow).

## Conventions

- **Build / preview:** `uv run --group docs zensical build --clean`.
- **Structure:** Diátaxis (tutorial / how-to / reference / explanation). A page
  may start as a combination of modes and be split later. Use an H1 title and H2
  sections.
- **Register a page** in both `zensical.toml` (`nav`) and the section index page
  (for example `docs/guide/index.md`).
- **Examples:** illustrative `@given` snippets use plain ` ```python ` fences
  and are not run; doctest examples use `>>>` and are collected via the pytest
  `--doctest-glob` setting. Verify every example against the real source and
  tests.
- **Cross-link, do not duplicate** prose across files (the README ↔
  `docs/index.md` problem): keep one source and link to it.
- **Reference** pages are generated from docstrings via `mkdocstrings` (see
  `zensical.toml`); their quality equals docstring quality.
- **Living pages** (for example bugs-found and roadmap) are reviewed each
  release.
