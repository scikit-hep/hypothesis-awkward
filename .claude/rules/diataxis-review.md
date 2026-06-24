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
- **Does NOT belong:** procedural steps and parameter how-to that _teach a task_
  — link to the how-to or reference instead. A short, clearly-labelled
  illustrative sketch that _depicts_ an idea (for example a non-runnable
  `@given` snippet) is fine; a runnable procedure that _instructs_ is not.
- **Failure to catch:** facts without the why; missing connective tissue; the
  page quietly turning into a how-to or a reference.

## Mixed pages

A page may target more than one quadrant **only when the page plan declares it**
(it is tagged with more than one quadrant). Then answer each assigned question,
and additionally judge whether the modes are kept distinct within the page or
muddled into each other. A page tagged with a single quadrant must stay in that
one.

## Self-check (run before reporting)

1. **Demand side.** Every point you raise must serve one of the page's assigned
   questions. If an ask would pull the page toward a quadrant it does **not**
   target, label it **out of scope** and route it to the page that owns that
   quadrant — do not report it as a defect of this page. In particular, do not
   demand that the page _acquire_ another quadrant's content it does not already
   have: do not ask an Explanation page to add runnable how-to steps or a worked
   example; that content belongs in the page you link to.
2. **Supply side.** Flag content **already on the page** that _teaches or
   instructs_ in another quadrant's mode as **bleed to relocate**, naming where
   it should go. But content that serves _this_ page's mode is earned even when
   it resembles another quadrant's form: a short, labelled, non-runnable
   `@given` sketch that depicts a concept on an Explanation page illustrates the
   idea rather than teaching a task — leave it. Distinguish depicting (earned)
   from instructing (bleed).

Report the result of both passes as the alignment self-check in your review.
