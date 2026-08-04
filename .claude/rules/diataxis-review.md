# Diátaxis review

How to review a document against the Diátaxis quadrant(s) its units of content
are written for. Used by the persona reviewers in the persona-review workflow.
Each unit of content declares its own quadrant with a marker in the document
itself; there is no separate table to keep in sync. The marker syntax, the
shapes a document may take, and the declaration record are defined in
`.claude/rules/diataxis-declaration.md` (the declaration file).

## How to use this

You will be told which quadrant each unit targets and asked the matching reader
question(s). Review each unit in that mode, applying the guidance below
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
| Tutorial    | "By following this unit, did you learn to do it yourself?"     | action + acquisition    |
| How-to      | "Using this unit, could you accomplish your task?"             | action + application    |
| Reference   | "Were you informed — could you find and trust the exact fact?" | cognition + application |
| Explanation | "After reading, do you understand it — does it make sense?"    | cognition + acquisition |

## Reviewing each quadrant

For each quadrant declared in the document, check three things: does the unit do
what the mode requires, does it keep out what the mode excludes, and does it
avoid the mode's characteristic failure.

### Tutorial — "Do I learn this?" (action + acquisition)

- **Must do:** a guaranteed-to-succeed guided lesson that is concrete,
  repeatable, and confidence-building; the unit takes responsibility for the
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
  the unit's work_ — link to the how-to or reference instead. (An illustrative
  example may appear if it stays subordinate to the explanation; it becomes out
  of quadrant once it starts doing the how-to's job — see the self-check.)
- **Failure to catch:** facts without the why; missing connective tissue; the
  unit quietly turning into a how-to or a reference.

## One quadrant per unit

Each unit does **one** job. Diátaxis allows complex _structure_, not blended
_content_: a document that must serve two needs gives each need its own
single-purpose section rather than letting the modes run together. No unit is
declared with two quadrants, and a section that blends modes is muddled, not a
sanctioned mix.

A run may restructure the document, and each operation is constrained so that no
unit ends up carrying two quadrants:

- **Create / remove** — a run may add the section a quadrant needs and remove
  one that no longer carries content. A new section carries a marker from the
  moment it is created, taken from the quadrant of the content moving into it.
  This is the destination for out-of-quadrant content that belongs in the
  document but has no unit to receive it, and it may only take content already
  in the document — a run never writes new material into another quadrant to
  fill it.
- **Split** — same-quadrant: the products inherit the original marker. A split
  never assigns a new quadrant.
- **Merge** — same-quadrant only. Sections of different quadrants cannot merge,
  because the merged section would carry two declarations; group them as
  subsections under a container heading instead, each keeping its own marker.
- **Relocate** — how out-of-quadrant content is fixed: it moves to a unit that
  owns its quadrant, possibly into a section newly added in the destination
  quadrant; it is never polished in place. A run relocates only within the
  document it is working on; content whose home is another document is reported
  to the user as a suggested move, never carried out.
- **Reclassify** — changing the marker on existing content, or giving unmarked
  existing content its first marker. This is a scoping decision, made when a run
  is scoped or by the user, and never a review outcome.

The declared quadrant is the fixed point of a review round: reviewers judge
content against the marker, never the reverse.

## Self-check (run before reporting)

1. **Demand side.** Answer each assigned reader question, and make every point
   you raise serve one of them. If an ask would pull a unit toward a quadrant it
   does **not** target, label it **out of scope** and route it to the unit that
   owns that quadrant, or — when no unit does — to the destination the profile's
   Declaration mechanism names; do not report it as a defect of this unit. In
   particular, do not demand that a unit _acquire_ another quadrant's content it
   does not already have: do not ask an Explanation unit to add runnable how-to
   steps or a worked example; that content belongs in the unit you link to.
2. **Supply side.** Cross-mode material may appear **only subordinate to the
   unit's one purpose**. The test is _service, not status_: does a passage still
   do _this_ unit's job, or has it begun doing another mode's? Flag content
   **already in the document** as **out-of-quadrant content to relocate**
   (naming where it goes) once it crosses that line — an illustrative code
   sketch that _depicts_ a concept serves an Explanation unit, but adjacent
   steps telling the reader how to run or narrow it _now_ are out-of-quadrant
   how-to content. (Diátaxis: illustrative examples are fine, but become
   out-of-quadrant content when they "develop into" the other mode and interrupt
   the unit's purpose.) Name the destination, taking the first that fits: a unit
   in this document that already owns the quadrant; failing that, a new section
   in this document carrying that quadrant; and if the content's home is another
   document, neither — leave it in place and report it as a cross-document move
   suggested to the user. A run restructures the document in front of it, never
   the document set, so content is never taken out of a document with nothing to
   receive it.
3. **Declaration.** List each unit you reviewed with its heading and the marker
   you read for it, quoting the marker verbatim so a malformed value is visible.
   Report as defects: a unit with no marker; a marker misplaced or malformed per
   the declaration file; a value outside the declaration file's legend; and any
   breach of the document shapes the declaration file defines. These are defects
   the run cannot fix itself, since giving existing content a marker is a
   scoping decision. If a marker's value does not match what the unit actually
   does, say so as a proposed reclassification, which is likewise scoping and
   not a review outcome. Apply any value-specific checks the declaration file
   defines.

Report the result of all three passes as the alignment self-check in your
review.
