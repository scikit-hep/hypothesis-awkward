# README Intro Rewrite via Persona Review

- **Date:** 2026-06-17
- **Status:** Done (branch `docs/readme-intro`, commit `11cf459`)

## Motivation

The README intro (the text between the badges and `## Installation` in
[README.md](../../README.md)) had two problems:

- It compared the package to NumPy / pandas / xarray / PyArrow strategies —
  noise that did not help a reader decide whether to use it.
- It described the package as generating "certain types of layouts," which
  **understated** reality:
  [`arrays()`](../../src/hypothesis_awkward/strategies/constructors/array_.py)
  already generates _nearly fully general_ Awkward Arrays.

Sharper framing (the combinatorial space of valid arrays, shrinking, current
capability) was lifted from the CHEP-2026 presentation script.

## Process overview

Rather than a single hand-edit, the intro was produced by a structured,
persona-driven workflow:

1. Itemize the requirements into a rubric.
2. Fix the link set.
3. Define reviewer personas.
4. Draft three structurally distinct versions against the same rubric.
5. Review all three with each persona, in parallel.
6. Synthesize a final from the consolidated review.

## 1. Rubric (criteria)

- **A. Content** — explain property-based vs example-based testing; what a
  _strategy_ is; the payoff of _shrinking_; why Awkward Array specifically
  benefits (a combinatorial space hand-written samples cannot cover); what the
  package is and is for; name `arrays()` as the entry point; state current
  capability honestly alongside the goal.
- **B. Accuracy** — "nearly fully general"; leaf dtypes = _the NumPy dtypes
  Awkward Array supports_; correct layout/feature names (nested,
  variable-length, record, union; optional/masked/missing; virtual); correct
  Hypothesis terms (strategies, shrinks, given); aim (fully general) ≠ present
  state.
- **C. Exclusions** — no NumPy/pandas/xarray/PyArrow comparison; drop "certain
  types of layouts"; no overclaim presenting the goal as done.
- **D. Editorial/mechanical** — low jargon load; tight altitude (node-level
  detail belongs to later sections / API ref); no contradiction with the
  downstream "current version generates…" paragraph; consistent "Awkward Array"
  naming; ~80-col hard wrap; MD040; no stray `>>>`; no dangling link references.

## 2. Links

Keep and use `[hypothesis]`, `[hyp-st]`, `[awkward-array]`,
`[hypothesis-awkward]`; link `arrays()` on first mention to `[api-ref-arrays]`
(defined later in the README — markdown reference definitions are
document-wide). Remove five now-orphaned refs: `hyp-st-numpy`, `hyp-st-pandas`,
`xarray`, `xarray-st`, `pyarrow-st`.

## 3. Reviewer personas

Four non-overlapping lenses:

- **New Adopter** — a physicist/RSE who uses Awkward Array but has never written
  a strategy. Owns comprehension, jargon load, "is this for me / what do I run?"
- **PBT Practitioner** — a Hypothesis expert. Owns terminology accuracy
  (strategies, shrinking, the example-vs-property framing).
- **Domain Expert** — an Awkward Array maintainer. Owns capability claims,
  verified against the source.
- **Docs Editor** — a scikit-hep PR reviewer. Owns prose, consistency, link
  hygiene, and the editorial consequences of what the edit removes.

## 4. Three versions

Three structurally distinct drafts, same rubric and links:

- **V1 — Definition-first:** Hypothesis → why Awkward Array → the package.
- **V2 — Problem-first:** the combinatorial-space pain → property-based testing
  as the answer → the package. (The presentation's narrative arc.)
- **V3 — Value-first:** what the package _is and does_ in sentence one →
  justify.

## 5. Parallel review and result

Each persona ran as an **independent parallel subagent**, scoring all three
versions through its lens. The Domain Expert fact-checked claims against
`array_.py` and the supported-dtype set
([`util/dtype.py`](../../src/hypothesis_awkward/util/dtype.py),
`SUPPORTED_DTYPES`).

| Persona          | Winner |
| ---------------- | ------ |
| New Adopter      | V3     |
| PBT Practitioner | V1     |
| Domain Expert    | V1     |
| Docs Editor      | V2     |

Borda count (3/2/1): **V1 = V2 = V3 = 8** — a perfect three-way tie. Each
version won on its own axis, which is the clearest signal that synthesis beats
picking one.

## 6. Synthesis and fixes

The final intro took **V2's problem → solution → package arc**, pulled in **V3's
"called with no arguments" `arrays()` framing**, and grafted the most-praised
sentences. Cross-cutting fixes applied regardless of version:

- "strategies are Python functions" → "composable objects that describe how to
  build test data" (PBT).
- property holds for "any **valid** input", not "generated input" (PBT).
- "the simplest sample" → "a minimal sample"; kept the "searching for … still
  triggers the failure" framing (shrinking is best-effort, not a guaranteed
  global minimum) (PBT).
- dropped "lazy evaluation" for virtual arrays — it is deferred _buffer
  materialization_ (Domain Expert). This overrode the Docs Editor's stylistic
  preference for the gloss: **accuracy beat style.**
- goal sentence → "full generality" to avoid repeating "fully general" (Editor).
- `[Hypothesis]` linked once; each proper noun linked on first mention (Editor).

One judgment call: **EmptyArray** was left out of the capability list (an
acceptable intro-level simplification; it is still named in the later "current
version generates…" paragraph).

A final manual tweak removed the word **"already"** — "nearly fully general"
plus "the goal is full generality" already carries the present-vs-aim contrast.

## Outcome

Applied to [README.md](../../README.md) only. Verified: no dangling refs, all
used labels resolve, ≤88-col wrap, no doctest prompts. Committed on branch
`docs/readme-intro` as `docs: rewrite README intro` (`11cf459`); prettier
reflowed the wrapping on the first attempt.

## Follow-ups

- The intro is duplicated nearly verbatim in
  [docs/index.md](../../docs/index.md) (intro paragraphs + shared link block);
  it was **not** synced in this pass and needs the same rewrite + ref pruning.

## Reusable workflow

The general pattern, for future substantial prose/doc changes:

1. **Rubric** — itemize what the text must say, must be true, must exclude, and
   must satisfy mechanically.
2. **Personas** — define a few non-overlapping reviewer lenses (comprehension,
   terminology, domain accuracy, editorial).
3. **Diverse drafts** — write a small number of _structurally distinct_ versions
   (3 is usually enough), all meeting the rubric, varying only framing.
4. **Parallel persona review** — score every version through every lens
   independently; consolidate into a matrix.
5. **Synthesize** — when scores tie or split by axis, merge the per-axis winners
   rather than picking one; apply cross-cutting fixes (accuracy beats style).
