---
name: write-docs-page
description:
  Author or substantially revise a docs page via the persona-review workflow
disable-model-invocation: false
---

Author (or substantially revise) a page under `docs/` using the narrative-track
persona-review workflow proven on the README intro and the _Testing Awkward
Array_, _Generating and Shrinking Samples_, and _Roadmap_ guide pages. The goal
is a page that serves its primary audience and is accurate; **accuracy beats
style**.

Strategy and the page backlog live in
`.design/notes/2026-06-17-02-Docs-plan.md`; the six review personas are the
`docs-persona-*` subagents in `.claude/agents/`. Voice rules are in
`rules/docs-voice.md`; operational conventions (build, fences, nav) are in
`rules/docs.md`.

## Steps

1. **Scope** — Confirm the page, its Diátaxis quadrant(s), and its primary
   audience from the page-plan table in the docs plan note, and note the
   matching reader question(s) for each quadrant (see
   `rules/diataxis-review.md`). For a new page, judge how well it fits as the
   next page to add.

2. **Gather sources** — Collect the raw material: in-repo evidence (tests,
   source), existing docs, and any external material the user points to.

3. **Rubric** — Itemize what the page must say (Content), must be true
   (Accuracy), must exclude (Exclusions), and must satisfy editorially
   (`rules/docs-voice.md`). For a page with no concrete code or tests to anchor
   it (explanation, meta, or forward-looking), pin the domain framing first —
   the conceptual model and the key terms and distinctions. All three drafts
   inherit the framing, so a wrong frame poisons them identically and the
   persona pass will not reliably catch it.

4. **Diverse drafts** — Write three structurally distinct drafts, all meeting
   the rubric, varying only the framing and order.

5. **Parallel persona review** — Launch the six persona subagents in
   `.claude/agents/` (`docs-persona-awkward-core-dev`,
   `docs-persona-downstream-dev`, `docs-persona-researcher`,
   `docs-persona-pbt-expert`, `docs-persona-evaluator`, `docs-persona-ai`) in
   parallel via the Agent tool's `subagent_type`. Write the drafts and a shared
   review brief (the page's purpose; its Diátaxis quadrant(s) and the matching
   reader question(s) from `rules/diataxis-review.md`; what is in and out of
   scope; rubric; verified facts; link targets) to temp files and pass each
   subagent their paths. Ask each for: an answer to each assigned quadrant's
   reader question; a score per draft on the rubric axes; lens-specific flags
   with quoted text and `file:line` citations; how relevant the page is to it;
   the best draft overall and per axis; specific fixes; the single most
   important improvement; an alignment self-check (which points are out of scope
   for the page's quadrant, and where they are routed); and a one-line
   ship/revise verdict (with the single most important change if revising).
   Consolidate into a matrix. If a reviewer errors out mid-run, re-launch it —
   do not treat a missing verdict as a pass. This pass validates lens-relevance
   and accuracy, not framing or altitude; the re-review step below covers that.

6. **Fact-check** — Verify every claim and code example against the actual
   source and tests (real API import paths, real upstream tests), and verify
   issue and pull-request titles and status on GitHub. When a claim summarizes
   an ordered list (for example a "shrinks toward" table), verify the full order
   against the source, not just the first item. Accuracy beats style.

7. **Synthesize or select** — First, if the review surfaced a flaw shared by all
   three drafts (most often a framing problem), fix it across the drafts and
   re-review once or twice before proceeding — the diverse drafts only help once
   the shared frame is right. Then produce the page: when strengths are split
   across drafts, merge the per-axis winners; when one draft is strongest on
   most axes, take it as the base and graft only the specific wins from the
   others. Merging adds seams, so do not merge for its own sake. Apply
   cross-cutting fixes; follow `rules/docs-voice.md`. Keep the page within its
   declared quadrant(s): an ask a persona flagged out of scope, and any on-page
   content flagged as bleed, is routed to the owning page's backlog in the docs
   plan note — not folded into this page.

8. **Re-review the resulting page** — The draft review (step 5) does not cover
   the page you will ship: a merge can inherit a weakness shared by all three
   drafts, and a chosen-and-edited draft carries changes no reviewer saw. Run
   the personas again on the resulting page as a single document (same brief,
   single-page critique), apply the genuine fixes, and re-review — iterating
   until all six return a "ship" verdict (cap at five rounds). Re-run after each
   edit round, since a fix can introduce a new error, and re-verify the build
   each round. Apply each persona's feedback within the page's Diátaxis quadrant
   (see Guidelines).

9. **Wire and verify** — Register the page in `zensical.toml` (`nav`) and the
   section index page. Run prettier, doctest collection, and `zensical build`,
   and confirm links resolve (see `rules/docs.md`).

10. **Update status** — Set the page's status in the page-plan table in the docs
    plan note.

## Guidelines

- Link to existing prose; do not duplicate it (the README ↔ `docs/index.md`
  problem).
- A page may be declared in the page plan as a combination of Diátaxis quadrants
  and split later as it grows; undeclared cross-quadrant content is bleed, not a
  sanctioned mix (see `rules/diataxis-review.md`).
- Personas review and route by quadrant per `rules/diataxis-review.md`: a lens
  asking for content outside the page's mode — for example runnable how-to steps
  on an explanation page — is out of scope, not a defect; route it (and any
  on-page bleed) to the page that owns that quadrant instead of folding it in.
- A page is written well when the personas it serves find it relevant, and the
  personas it does not serve can tell early that it is not for them while still
  seeing it is useful to other readers.
- Do not hand-type "last updated" dates; see the docs plan's Open questions
  (pending Zensical revisioning).
