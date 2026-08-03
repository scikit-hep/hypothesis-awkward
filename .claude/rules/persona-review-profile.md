# Persona-review profile

Repository-specific values for the persona-review engine in
`.claude/skills/write-docs-page/SKILL.md`. The engine reads this file first and
refers to its sections by name; the section structure is shared with the
counterpart profile in legendary-octo-happiness.

## Document

The document is one page under `docs/`; the page set grows page by page.
Strategy and the page backlog live in `.design/notes/2026-06-17-02-Docs-plan.md`
(the docs plan note). The section set is not an output of the run: the page is
the unit of work, and drafts vary only framing and order. Scoping note: for a
new page, judge how well it fits as the next page to add. Provenance: this is
the narrative-track workflow, proven on the README intro and the _Testing
Awkward Array_, _Generating and Shrinking Samples_, and _Roadmap_ guide pages.

## Personas

The six review personas are the `docs-persona-*` subagents in `.claude/agents/`:
`awkward-core-dev`, `downstream-dev`, `researcher`, `pbt-expert`, `evaluator`,
and `ai`. The primary personas are the page's primary audience in the page-plan
table in the docs plan note.

## Declaration mechanism

Each page's Diátaxis quadrant(s) and primary audience are declared in the
page-plan table in the docs plan note. The Diátaxis rules are
`.claude/rules/diataxis-review.md`. A page may be declared as a combination of
quadrants and split later as it grows; undeclared cross-quadrant content is out
of quadrant, not a sanctioned mix. Declarations reach reviewers via the review
brief, which carries the page's quadrant(s) and the matching reader question(s).
Out-of-scope asks and out-of-quadrant content are routed to the owning page's
backlog in the docs plan note.

## Premise to pin

Trigger: a page with no concrete code or tests to anchor it (explanation, meta,
or forward-looking). Premise: the domain framing — the conceptual model and the
key terms and distinctions.

## Sources

In-repo evidence (tests, source); existing docs; and any external material the
user points to.

## Fact-check targets

The actual source and tests: real API import paths, real upstream tests. Issue
and pull-request titles and status on GitHub. Checking note: when a claim
summarizes an ordered list (for example a "shrinks toward" table), verify the
full order against the source, not just the first item.

## Status dimension

Disabled.

## Verification

One-time wiring: register the page in `zensical.toml` (`nav`) and the section
index page. Checks, re-run each review round: prettier; doctest collection;
`zensical build`; and confirm links resolve. Operational conventions (build,
fences, nav) are in `.claude/rules/docs.md`.

## Record

Set the page's status in the page-plan table in the docs plan note.

## Voice rules

`.claude/rules/docs-voice.md`.

## Extra guidelines

- Link to existing prose; do not duplicate it (the README ↔ `docs/index.md`
  problem).
- Do not hand-type "last updated" dates; see the docs plan's Open questions
  (pending Zensical revisioning).
