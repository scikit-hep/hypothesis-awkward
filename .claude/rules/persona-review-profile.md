# Persona-review profile

Repository-specific values for the persona-review engine in
`.claude/skills/write-docs-page/SKILL.md`. The engine reads this file first and
refers to its sections by name; the section structure is shared with the
counterpart profile in legendary-octo-happiness.

## Document

The document is one page under `docs/`; the page set grows page by page.
Strategy and the page backlog live in `.design/notes/2026-06-17-02-Docs-plan.md`
(the docs plan note); per-page primary audience and update trigger live in
`.design/docs/Chapters.md` (the chapters table). The page is the unit of work,
but the unit of declaration is finer: a page is either one unit or a container
of single-purpose sections, each carrying its own marker. The section set is an
output of the run — a run may create, remove, split, merge, and relocate
sections under the rules in `.claude/rules/diataxis-review.md`. Scoping note:
for a new page, judge how well it fits as the next page to add. Provenance: this
is the narrative-track workflow, proven on the README intro and the _Testing
Awkward Array_, _Generating and Shrinking Samples_, and _Roadmap_ guide pages.

## Personas

The six review personas are the `docs-persona-*` subagents in `.claude/agents/`:
`awkward-core-dev`, `downstream-dev`, `researcher`, `pbt-expert`, `evaluator`,
and `ai`. The primary personas are the page's primary audience in the chapters
table; for a page that does not exist yet, its row in the page backlog in the
docs plan note.

## Declaration mechanism

Each unit's Diátaxis quadrant is declared by a marker in the page itself — an
HTML comment directly below the unit's heading. The Diátaxis rules are
`.claude/rules/diataxis-review.md`, which holds the marker syntax, the two page
shapes (one unit, or a container of marked sections), and the rules for
restructuring. The markers are the record; there is no separate table of
quadrants. No unit is declared with two quadrants. Content that belongs to no
unit is out of quadrant by construction, with one sanctioned exception: the
orientation preamble a container page may carry between its H1 and its first H2.

Declarations travel with the text: each draft carries its own markers, so the
declaration under review is always the one in the draft. The review brief
repeats each unit's quadrant and the matching reader question, and gives the
page's primary audience from the chapters table. Out-of-quadrant content moves
only within the page under review, into a unit that owns its quadrant or into a
new section carrying it. An out-of-scope ask, or content whose home is another
page, is reported as a suggestion and recorded in the routed-items list under
**Page backlog** in the docs plan note; acting on it is the user's decision,
since a run restructures one page and never the page set. Reclassifying existing
content — or giving unmarked existing content its first marker — is likewise a
scoping decision, not a review outcome.

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
`zensical build`; and confirm links resolve. The markers are checked by the
reviewers' declaration pass, not mechanically. Operational conventions (build,
fences, nav) are in `.claude/rules/docs.md`.

## Record

The markers in the shipped page are the record of its quadrants, so no separate
update is needed. For a page that did not exist before, add its row to the
chapters table and remove its row from the page backlog in the docs plan note.

## Voice rules

`.claude/rules/docs-voice.md`.

## Extra guidelines

- Link to existing prose; do not duplicate it (the README ↔ `docs/index.md`
  problem).
- Do not hand-type "last updated" dates; see the docs plan's Open questions
  (pending Zensical revisioning).
