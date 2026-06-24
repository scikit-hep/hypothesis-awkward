# Diátaxis Review

How to review a documentation page against the Diátaxis quadrant(s) it is
written for. Used by the `docs-persona-*` reviewers in the `write-docs-page`
workflow. A page's quadrant assignment comes from the page-plan table in
`.design/notes/2026-06-17-02-Docs-plan.md`.

## How to use this

You will be told which quadrant(s) the page targets and asked the matching
reader question(s). Review the page in that mode, applying the guidance below
**through your own persona lens** (your audience, pain points, and what you
value still hold — but only to the extent the assigned quadrant calls for them).
Before reporting, run the self-check at the end.

Diátaxis places each kind of documentation on two axes:

- **action ↔ cognition** — practical doing versus theoretical knowing.
- **acquisition ↔ application** — studying (learning) versus working (a task in
  hand).

The four quadrants and the question their reader arrives with:

| Quadrant    | Reader question                                                | Axes                    |
| ----------- | -------------------------------------------------------------- | ----------------------- |
| Tutorial    | "By following this page, did you learn to do it yourself?"     | action + acquisition    |
| How-to      | "Using this page, could you accomplish your task?"             | action + application    |
| Reference   | "Were you informed — could you find and trust the exact fact?" | cognition + application |
| Explanation | "After reading, do you understand it — does it make sense?"    | cognition + acquisition |

## Reviewing each quadrant

For each quadrant the page is assigned, check three things: does it do what the
mode requires, does it keep out what the mode excludes, and does it avoid the
mode's characteristic failure.

### Tutorial — "Do I learn this?" (action + acquisition)

- **Must do:** a guaranteed-to-succeed guided lesson that is concrete,
  repeatable, and confidence-building; the page takes responsibility for the
  learner reaching the end; doing comes first and meaning later.
- **Does NOT belong:** options, forks, and alternatives; theory and rationale;
  unexplained prerequisites; anything that leaves the learner to figure it out.
- **Failure to catch:** assumed competence — a missing first step, an undefined
  term, or a branch that strands a true beginner.

### How-to — "Can I do this for my work?" (action + application)

- **Must do:** a reliable sequence of actions that achieves a real-world goal
  for a reader who is already competent.
- **Does NOT belong:** teaching from scratch, conceptual digression,
  completeness for its own sake (it serves one task, not every task).
- **Failure to catch:** steps that do not actually reach the goal — gaps,
  detours, or a wrong assumed starting point.

### Reference — "Am I informed?" (cognition + application)

- **Must do:** an accurate, complete, consistently structured description of the
  machinery (every parameter, default, return, edge case) that mirrors the
  structure of the thing described and is quick to scan.
- **Does NOT belong:** instructions, persuasion or justification, narrative that
  buries the facts.
- **Failure to catch:** anything missing, stale, or inconsistent; a fact that is
  hard to find or cannot be trusted.

### Explanation — "Do I understand this?" (cognition + acquisition)

- **Must do:** build a mental model — the why, the connections, the design
  rationale, trade-offs, alternatives, and the boundaries of the idea; it may
  hold opinion and discussion; it is understood away from the keyboard.
- **Does NOT belong:** procedural steps, worked how-to, or parameter lookup _as
  the page's work_ — link to the how-to or reference instead. (An illustrative
  example may appear if it stays subordinate to the explanation; it becomes
  bleed once it starts doing the how-to's job — see the self-check.)
- **Failure to catch:** facts without the why; missing connective tissue; the
  page quietly turning into a how-to or a reference.

## Mixed pages

Each page — and each section within it — does **one** job. Diátaxis allows
complex _structure_, not blended _content_: a page that must serve two needs
gives each its own single-purpose section, or is split, rather than letting the
modes run together. When the page plan tags a page with more than one quadrant,
read that as distinct single-purpose sections — answer each section's question
and check that each stays in its lane. A section that blends modes is muddled,
not a sanctioned mix; a page tagged with a single quadrant must stay in that
one.

## Self-check (run before reporting)

1. **Demand side.** Every point you raise must serve one of the page's assigned
   questions. If an ask would pull the page toward a quadrant it does **not**
   target, label it **out of scope** and route it to the page that owns that
   quadrant — do not report it as a defect of this page. In particular, do not
   demand that the page _acquire_ another quadrant's content it does not already
   have: do not ask an Explanation page to add runnable how-to steps or a worked
   example; that content belongs in the page you link to.
2. **Supply side.** Cross-mode material may appear **only subordinate to this
   page's one purpose**. The test is _service, not status_: does a passage still
   do _this_ page's job, or has it begun doing another mode's? Flag content
   **already on the page** as **bleed to relocate** (naming where it goes) once
   it crosses that line — an illustrative `@given` sketch that _depicts_ a
   concept serves an Explanation page, but adjacent steps telling the reader how
   to run or narrow it _now_ are how-to bleed. (Diátaxis: illustrative examples
   are fine, but become bleed when they "develop into" the other mode and
   interrupt the page's purpose.)

Report the result of both passes as the alignment self-check in your review.
