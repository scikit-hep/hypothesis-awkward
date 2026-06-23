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
  - In a runnable Markdown doctest, leave a blank line before the closing code
    fence, or the default doctest parser captures the fence line as part of the
    expected output.
  - Prefer a deterministic example, for example
    `find(strategy, predicate, settings=settings(database=None))`, and avoid
    `.example()` (non-deterministic); confirm the exact repr by running it.
- **Cross-link, do not duplicate** prose across files (the README ↔
  `docs/index.md` problem): keep one source and link to it.
- **Pages need not stand alone.** A page may assume concepts covered elsewhere,
  as long as it links to the introductory page or external reference where they
  are explained; match each page to its audience instead of re-teaching
  fundamentals everywhere.
- **Reference** pages are generated from docstrings via `mkdocstrings` (see
  `zensical.toml`); their quality equals docstring quality.
- **Living pages** (for example bugs-found and roadmap) are reviewed each
  release.
