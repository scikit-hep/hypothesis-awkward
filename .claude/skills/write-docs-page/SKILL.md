---
name: write-docs-page
description:
  Author or substantially revise a docs page via the persona-review workflow
disable-model-invocation: false
---

Every repository-specific value lives in this repository's
`.claude/rules/persona-review-profile.md` (the profile); read it first. The
profile supplies, by section: Document, Personas, Declaration mechanism, Premise
to pin, Sources, Fact-check targets, Status dimension, Verification, Record,
Voice rules, and Extra guidelines. In this file, "the Diátaxis rules" refers to
the rules file named in the profile's Declaration mechanism section, and "the
voice rules" to the file named in its Voice rules section.

Author (or substantially revise) the document defined in the profile's Document
section using the persona-review workflow. A revision may be triggered by one
change or one weak part, but drafting, review, and shipping cover the document
as a whole. When the profile's Document section declares the **section set an
output of the run**, sections are added, split, merged, and removed as the
content requires. The goal is a document whose content serves its primary
personas and is accurate; **accuracy beats style**. The review personas are the
subagents listed in the profile's Personas section.

Every unit of content is declared in a [Diátaxis](https://diataxis.fr/) quadrant
(tutorial, how-to, reference, or explanation) as the profile's Declaration
mechanism section directs; the Diátaxis rules hold the reader questions and the
out-of-quadrant and out-of-scope rules. When the profile's Status dimension
section is enabled, every unit also carries a **status** — _implemented_
(describes current behavior; verified against the profile's Fact-check targets)
or _spec_ (describes intended behavior; source of truth is a design brief
supplied when the skill is invoked — a decision list or note that need not live
in the repository).

## Steps

1. **Scope** — Confirm the change driving the revision, the unit of work, and
   what is out of scope; confirm the primary personas (whose verdicts outweigh
   the others when fixes conflict) as the profile's Personas section directs;
   and confirm the declared quadrant(s) per the Declaration mechanism, noting
   the matching reader question(s) from the Diátaxis rules. Apply any scoping
   notes in the profile's Document section. When the section set is an output of
   the run, sketch the target section set — starting from the current
   declarations, adding, splitting, merging, or removing sections as the content
   requires — and declare each section's quadrant. When the status dimension is
   enabled, confirm the change's status; for spec status, capture the design
   decisions the text must encode into the review brief (step 5) so the brief is
   self-contained.

2. **Gather sources** — Collect the raw material listed in the profile's Sources
   section.

3. **Rubric** — Itemize what the document must say (Content), must be true
   (Accuracy), must exclude (Exclusions), and must satisfy editorially (the
   voice rules). When the trigger named in the profile's Premise to pin section
   applies, pin the named premise first. All three drafts inherit the premise,
   so a wrong one poisons them identically and the persona pass will not
   reliably catch it; settle the premise before drafting, against the authority
   the profile names, if it names one.

4. **Diverse drafts** — Write three structurally distinct drafts of the whole
   document, all meeting the rubric, to temp files. When the section set is an
   output of the run, structure is part of the variation: drafts may differ in
   how many sections exist and how content is distributed among them, as long as
   each draft keeps its declarations valid per the Declaration mechanism and
   declares its own structure. Otherwise, vary only the framing and order.

5. **Parallel persona review** — Launch the persona subagents listed in the
   profile's Personas section in parallel via the Agent tool's `subagent_type`.
   Write a shared review brief — the project and document identity, from the
   profile's Document section; the document's purpose; the declared quadrant(s)
   and matching reader question(s), carried as the Declaration mechanism
   directs; when the status dimension is enabled, each unit's status; what is in
   and out of scope; rubric; verified facts; link targets — to a temp file and
   pass each subagent its path plus the draft paths. Subagents never depend on
   files outside the repository and the brief; when the status dimension is
   enabled, the brief itself carries the design decisions for spec content. Ask
   each for: answers to the reader questions for the content its lens serves; a
   score per draft on the rubric axes; lens-specific flags with quoted text and
   `file:line` citations; how relevant the document is to it (per section, when
   the section set is an output of the run); the best draft overall and per
   axis; specific fixes; structural recommendations (sections to add, split,
   merge, or remove) when the section set is an output of the run; the single
   most important improvement; an alignment self-check per the Diátaxis rules
   (out-of-scope asks and their routing; out-of-quadrant content flagged); and a
   one-line ship/revise verdict (with the single most important change if
   revising). Consolidate into a matrix. If a reviewer errors out mid-run,
   re-launch it — do not treat a missing verdict as a pass. This pass validates
   lens-relevance and accuracy, not framing or altitude; the re-review step
   below covers that.

6. **Fact-check** — Verify every claim and code example against the targets in
   the profile's Fact-check targets section, applying its checking notes. When
   the status dimension is enabled, verify spec content against the design
   decisions in the brief, and verify implementability on the platform the
   profile names: a behavior the platform cannot deliver as written is a
   blocking defect. Accuracy beats style.

7. **Synthesize or select** — First, if the review surfaced a flaw shared by all
   three drafts (most often a flaw in the pinned premise), fix it across the
   drafts and re-review once or twice before proceeding — the diverse drafts
   only help once the shared premise is right. Then produce the document: when
   strengths are split across drafts, merge the per-axis winners; when one draft
   is strongest on most axes, take it as the base and graft only the specific
   wins from the others. Merging adds seams, so do not merge for its own sake.
   Apply cross-cutting fixes and write the final text yourself, following the
   voice rules — persona-suggested wording is advisory. An ask a persona flagged
   out of scope, and any content flagged as out of quadrant, is routed to the
   destination named in the profile's Declaration mechanism section — not folded
   in where it does not belong.

8. **Re-review the resulting document** — The draft review (step 5) does not
   cover the text you will ship: a merge can inherit a weakness shared by all
   three drafts, and a chosen-and-edited draft carries changes no reviewer saw.
   Run the personas again on the resulting document (same brief; the
   declarations travel with the text as the Declaration mechanism directs,
   however much a round has changed), apply the genuine fixes within the
   declared quadrant(s), and re-review — iterating until every persona returns a
   "ship" verdict (cap at five rounds). Re-run the checks in the profile's
   Verification section each round, since a fix can introduce a new error. If
   the cap is reached with dissent remaining, stop and present the unresolved
   verdicts to the user — do not keep bending the text to chase the last
   holdout.

9. **Verify** — Work through the profile's Verification section: perform any
   one-time wiring it lists, then run its checks.

10. **Record** — Record the run as the profile's Record section directs. When
    the status dimension is enabled, list the claims that describe intended
    behavior in the implementation plan, so each is re-verified against the
    shipped implementation.

## Guidelines

- A unit of content is written well when its primary personas find what they
  need and the others can tell early that it is not for them while still seeing
  it is useful to its own readers.
- Content is not obligated to serve every persona, and the document does not owe
  any persona content. The correct review from a low-relevance persona is a low
  relevance score and a ship verdict — not asks that bend the document toward
  its lens. When personas' fixes conflict, the primary personas from step 1 win.
- When the section set is an output of the run: relocating out-of-quadrant
  content, creating the section a quadrant needs, and removing a section that no
  longer serves anyone are actions the run takes, guided by persona feedback.
  Removal has exactly two legitimate sources: a persona speaking as the
  section's own audience (duplication, void purpose, vanished subject), or the
  consolidated matrix showing a section every persona finds low-relevance — the
  latter is the orchestrator's judgment at synthesis, never a single
  low-relevance persona's ask. Every section-set change is listed in the report;
  an ask the run chooses not to serve is reported with a keep/drop
  recommendation for the user.
- When the status dimension is enabled: the design decisions in the brief are
  settled for spec content. A persona ask that would change a decision is design
  feedback — surface it in the report for the user to rule on; never fold it
  into the text as if settled. A spec unit binds the implementation: after the
  implementation ships, a difference between behavior and the text is either an
  implementation bug or a change that re-enters this skill — never a silent doc
  drift.
- Declarations, their granularity, and what counts as a sanctioned combination
  of quadrants follow the profile's Declaration mechanism section; undeclared
  cross-quadrant content is out of quadrant. Personas review and route by
  quadrant per the Diátaxis rules: a lens asking for content outside a unit's
  declared mode — for example runnable how-to steps in explanation content — is
  out of scope, not a defect; route it (and any out-of-quadrant content) to the
  destination the profile names instead of folding it in. Out-of-quadrant
  content is relocated or routed, never polished in place.
- Voice and formatting follow the voice rules; the orchestrator writes the final
  text, not the personas.
- Apply the additional guidelines in the profile's Extra guidelines section.
