# Diátaxis Review

How to review a documentation page against the Diátaxis quadrant(s) its units
are written for. Used by the `docs-persona-*` reviewers in the `write-docs-page`
workflow. Each unit of content declares its own quadrant with a marker in the
page (see [Markers](#markers)); there is no separate table to keep in sync.

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
| Tutorial    | "By following this page, did you learn to do it yourself?"     | action + acquisition    |
| How-to      | "Using this page, could you accomplish your task?"             | action + application    |
| Reference   | "Were you informed — could you find and trust the exact fact?" | cognition + application |
| Explanation | "After reading, do you understand it — does it make sense?"    | cognition + acquisition |

## Markers

A unit of content declares its quadrant with an HTML comment placed directly
below the unit's heading, separated by one blank line:

```markdown
## Bugs found

<!-- diataxis: reference -->
```

The value is one of `tutorial`, `how-to`, `reference`, `explanation`, or
`none (<reason>)` for a page that is not a Diátaxis unit, such as a navigation
index. Markers are invisible to site readers: they do not change heading
anchors, the table of contents, or the search index.

A page takes one of two shapes:

- **One unit** — a single marker below the H1, and no other marker on the page.
- **Container** — no marker below the H1, and one below every H2. The material
  between the H1 and the first H2 is an **orientation preamble**: framing,
  audience, scope, and links out. It carries no marker and is not a unit; once
  it starts doing a quadrant's work, it has to become a marked section.

Markers sit at one level per branch: a heading whose subsections carry their own
markers is a container and carries none. Every page under `docs/` carries at
least one marker. Nothing enforces this mechanically: a unit with no marker
gives its reviewers no question to answer, so it surfaces in the declaration
pass of the self-check. Note what that covers — the page under review, when a
run happens. A page edited outside a run keeps whatever markers it has until its
next review.

A one-unit page that grows a section of another quadrant changes shape: the H1
marker is removed and every H2 gets one. That is a scoping decision, not
something a review does in passing.

## Reviewing each quadrant

For each quadrant declared on the page, check three things: does the unit do
what the mode requires, does it keep out what the mode excludes, and does it
avoid the mode's characteristic failure.

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
  the unit's work_ — link to the how-to or reference instead. (An illustrative
  example may appear if it stays subordinate to the explanation; it becomes out
  of quadrant once it starts doing the how-to's job — see the self-check.)
- **Failure to catch:** facts without the why; missing connective tissue; the
  unit quietly turning into a how-to or a reference.

## One quadrant per unit

Each unit does **one** job. Diátaxis allows complex _structure_, not blended
_content_: a page that must serve two needs gives each need its own
single-purpose section rather than letting the modes run together. No unit is
declared with two quadrants, and a section that blends modes is muddled, not a
sanctioned mix.

A run may restructure a page, and each operation is constrained so that no unit
ends up carrying two quadrants:

- **Create / remove** — a run may add the section a quadrant needs and remove
  one that no longer carries content. A new section carries a marker from the
  moment it is created, taken from the quadrant of the content moving into it.
- **Split** — same-quadrant: the products inherit the original marker. A split
  never assigns a new quadrant.
- **Merge** — same-quadrant only. Sections of different quadrants cannot merge,
  because the merged section would carry two declarations; group them as
  subsections under a container heading instead, each keeping its own marker.
- **Relocate** — how content crosses quadrants, possibly into a section newly
  added in the destination quadrant.
- **Reclassify** — changing the marker on existing content, or giving unmarked
  existing content its first marker. This is a scoping decision, made when a run
  is scoped or by the user, and never a review outcome.

The declared quadrant is the fixed point of a review round: reviewers judge
content against the marker, never the reverse.

## Self-check (run before reporting)

1. **Demand side.** Every point you raise must serve one of the assigned
   questions. If an ask would pull a unit toward a quadrant it does **not**
   target, label it **out of scope** and route it to the unit that owns that
   quadrant, or — when no unit does — to the destination the profile's
   Declaration mechanism names; do not report it as a defect of this page. In
   particular, do not demand that a unit _acquire_ another quadrant's content it
   does not already have: do not ask an Explanation unit to add runnable how-to
   steps or a worked example; that content belongs in the page you link to.
2. **Supply side.** Cross-mode material may appear **only subordinate to the
   unit's one purpose**. The test is _service, not status_: does a passage still
   do _this_ unit's job, or has it begun doing another mode's? Flag content
   **already on the page** as **out-of-quadrant content to relocate** (naming
   where it goes) once it crosses that line — an illustrative `@given` sketch
   that _depicts_ a concept serves an Explanation unit, but adjacent steps
   telling the reader how to run or narrow it _now_ are out-of-quadrant how-to
   content. (Diátaxis: illustrative examples are fine, but become
   out-of-quadrant content when they "develop into" the other mode and interrupt
   the page's purpose.) Flag it for relocation only when the destination unit
   exists. When it does not, the content stays where it is and you record the
   intent in the backlog instead — otherwise a page loses material to a list in
   a design note and no page receives it.

3. **Declaration.** List each unit you reviewed with its heading and the marker
   you read below that heading, quoting the marker line verbatim so a misspelled
   value is visible. Report as defects: a unit with no marker; a marker that is
   not directly below its heading; a value outside the allowed set; and any
   breach of the page's shape — markers at more than one level of a branch, a
   container with an unmarked H2, or an H1 marker alongside section markers.
   These are defects the run cannot fix itself, since giving existing content a
   marker is a scoping decision (below). If a marker's value does not match what
   the unit actually does, say so as a proposed reclassification, which is
   likewise scoping and not a review outcome. A unit marked `none (<reason>)`
   has no reader question: check only that the reason still holds and that the
   unit has not started doing a quadrant's work.

Report the result of all three passes as the alignment self-check in your
review.
