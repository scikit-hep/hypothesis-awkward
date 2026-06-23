---
name: write-docs-page
description:
  Author or substantially revise a docs page via the persona-review workflow
disable-model-invocation: false
---

Author (or substantially revise) a page under `docs/` using the narrative-track
persona-review workflow proven on the README intro and the _Testing Awkward
Array_ page. The goal is a page that serves its primary audience and is
accurate; **accuracy beats style**.

Strategy and the page backlog live in
`.design/notes/2026-06-17-02-Docs-plan.md`; the six review personas are the
`docs-persona-*` subagents in `.claude/agents/`. Voice rules are in
`rules/docs-voice.md`; operational conventions (build, fences, nav) are in
`rules/docs.md`.

## Steps

1. **Scope** — Confirm the page, its Diátaxis quadrant, and its primary audience
   from the page-plan table in the docs plan note. For a new page, judge how
   well it fits as the next page to add.

2. **Gather sources** — Collect the raw material: in-repo evidence (tests,
   source), existing docs, and any external material the user points to.

3. **Rubric** — Itemize what the page must say (Content), must be true
   (Accuracy), must exclude (Exclusions), and must satisfy editorially
   (`rules/docs-voice.md`).

4. **Diverse drafts** — Write three structurally distinct drafts, all meeting
   the rubric, varying only the framing and order.

5. **Parallel persona review** — Launch the six persona subagents in
   `.claude/agents/` (`docs-persona-awkward-core-dev`,
   `docs-persona-downstream-dev`, `docs-persona-researcher`,
   `docs-persona-pbt-expert`, `docs-persona-evaluator`, `docs-persona-ai`) in
   parallel via the Agent tool's `subagent_type`. Write the drafts and a shared
   review brief (rubric, verified facts, link targets) to temp files and pass
   each subagent their paths. Ask each for: a score per draft on the rubric
   axes; lens-specific flags with quoted text and `file:line` citations; how
   relevant the page is to it; the best draft overall and per axis; specific
   fixes; and the single most important improvement. Consolidate into a matrix.

6. **Fact-check** — Verify every claim and code example against the actual
   source and tests (real API import paths, real upstream tests), and verify
   issue and pull-request titles and status on GitHub. When a claim summarizes
   an ordered list (for example a "shrinks toward" table), verify the full order
   against the source, not just the first item. Accuracy beats style.

7. **Synthesize** — Merge the per-axis winners rather than picking one draft;
   apply cross-cutting fixes; follow `rules/docs-voice.md`.

8. **Wire and verify** — Register the page in `zensical.toml` (`nav`) and the
   section index page. Run prettier, doctest collection, and `zensical build`,
   and confirm links resolve (see `rules/docs.md`).

9. **Update status** — Set the page's status in the page-plan table in the docs
   plan note.

## Guidelines

- Link to existing prose; do not duplicate it (the README ↔ `docs/index.md`
  problem).
- A page may start as a combination of Diátaxis modes and be split later as it
  grows.
- A page is written well when the personas it serves find it relevant, and the
  personas it does not serve can tell early that it is not for them while still
  seeing it is useful to other readers.
- Do not hand-type "last updated" dates; see the docs plan's Open questions
  (pending Zensical revisioning).
