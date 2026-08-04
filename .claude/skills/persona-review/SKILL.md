---
name: persona-review
description:
  Run one round of persona review on a document or its drafts, as the
  repository's profile directs
---

Run one round of persona review: fixed-persona reviewers, each in its own
subagent, review a document (or several drafts of it), and their reviews are
collected and consolidated for the caller — an authoring workflow or a direct
invocation.

Every repository-specific value lives in the repository's
`.claude/rules/persona-review-profile.md` (the profile); read it first. The
Diátaxis rules are the shared review core at `references/diataxis-review.md` in
this skill's directory; read it too — composing the brief needs its reader
questions.

## Steps

1. **Determine the review request.** When invoked from an authoring run, the run
   state in the conversation supplies the draft path(s), the document's purpose,
   what is in and out of scope, the declared quadrant(s), each unit's status
   (when the profile's Status dimension section is enabled), the rubric, the
   verified facts and link targets, and the personas chosen for the run. When
   invoked standalone, default to: the shipped document named in the profile's
   Document section, as it stands; all personas listed in the profile's Personas
   section; implemented status, when the status dimension is enabled; no rubric.

2. **Compose the review brief** to a temp file: the project and document
   identity, from the profile's Document section; the document's purpose; the
   declared quadrant(s) and matching reader question(s), carried as the
   profile's Declaration mechanism section directs; when the status dimension is
   enabled, each unit's status, and the design decisions for spec content — the
   brief is self-contained; what is in and out of scope; the rubric, when the
   request has one; verified facts; link targets; and the path of the Diátaxis
   core, so reviewers read the core from the brief. Reviewers never depend on
   files outside the repository, this skill, and the brief.

3. **Launch the panel.** For each persona in the request, launch one
   `persona-reviewer` subagent via the Agent tool's `subagent_type`, all in
   parallel. Open each task prompt with the full content of that persona's head
   file, from the profile's Personas section, then give the brief path and the
   draft path(s). Ask each for: answers to the reader questions for the content
   its lens serves; a score per draft on the rubric axes, when the brief carries
   a rubric; lens-specific flags with quoted text and `file:line` citations; how
   relevant the document is to it (per section, when the section set is an
   output of the run); the best draft overall and per axis, when several drafts
   are under review; specific fixes; structural recommendations (sections to
   add, split, merge, or remove) when the section set is an output of the run;
   the single most important improvement; an alignment self-check per the
   Diátaxis rules (out-of-scope asks and their routing; out-of-quadrant content
   flagged); and a one-line ship/revise verdict (with the single most important
   change if revising).

4. **Collect and consolidate.** If a reviewer errors out mid-run, re-launch it —
   do not treat a missing verdict as a pass. Consolidate the reviews into a
   matrix and report it to the caller with each reviewer's verdict and flags.
