# Documentation Strategy

- **Date:** 2026-06-17
- **Status:** Draft (brainstorm kickoff)

A kickoff note for a documentation initiative. It collects the raw material from
an initial brainstorm: goals, the current state, tooling findings, exemplar docs
to learn from, a candidate update workflow, and a backlog of pages. It is
deliberately one consolidated note; sections will be split out later (see
[Split-later triggers](#split-later-triggers)).

## Goals

- **Useful** — serves doers _and_ readers. A doer can find how to do their task
  and copy runnable code; a reader who never runs anything can still understand
  what the package is for and why it matters — e.g., how it makes Awkward
  Array's test suite more reliable and what bugs it has surfaced.
- **Frequently updated** — low risk to change. Examples are tested, prose is not
  duplicated, and updates ride the feature/release cycle.
- **Scalable** — grows without linear effort. Reference is generated from
  docstrings; durable link targets (a glossary) keep prose short; pages share
  predictable shapes.

## Current state

- `docs/guide/`: Overview, Installation, Getting Started — thin (install + a
  first use), no how-to or explanation layer.
- **Auto-generated API reference** via `mkdocstrings-python`, bridged into
  Zensical through [`zensical.toml`](../../zensical.toml)
  (`[project.plugins.mkdocstrings.handlers.python]`). Reference pages are
  one-line directives:

  ```markdown
  ::: hypothesis_awkward.strategies.constructors
  ```

  Docstrings drive everything (`docstring_style = "auto"`, NumPy default;
  `members_order = ["__all__", "source"]`; summary tables on). Cross-references
  resolve into Python / Hypothesis / NumPy / Awkward via configured
  `inventories`.

- **Versioning** via the `mike` fork (`provider = "mike"`, `dev` + `latest`) —
  see [`.claude/rules/ci.md`](../../.claude/rules/ci.md).
- **Tested examples**: doctest is enabled on docstrings and Markdown.
- **Symptom to fix**: the README intro is duplicated nearly verbatim in
  [`docs/index.md`](../../docs/index.md). Hand-written prose rots and duplicates
  while the generated reference never does. See
  [`2026-06-17-readme-intro-persona-review.md`](./2026-06-17-readme-intro-persona-review.md).

## Tooling findings (Zensical)

- Built by the **Material for MkDocs team**; explicitly **alpha**, targeting
  feature parity with Material for MkDocs.
- **Native versioning and API docs are roadmapped, not shipped** — which is why
  the project uses the `mike` fork and bridges in `mkdocstrings` today.
- Search ("Disco") is available; plugins are being replaced by a module system
  (Python bindings via PyO3).
- **Implication — decouple content from renderer.** Keep docs as portable,
  Diátaxis-structured Markdown; keep examples tested by our own pytest/doctest
  (renderer-independent); treat Zensical as a swappable presentation layer. This
  protects "scalable" and "frequently updated" against tooling churn (already
  felt via the `mike` fork).

## Exemplars

Docs worth learning from, split by what they model. Structure/content models are
renderer-agnostic; tooling cousins matter because Zensical targets Material for
MkDocs parity.

| Model                                           | Why it's good                                                             | Transferable bit                          | Kind      |
| ----------------------------------------------- | ------------------------------------------------------------------------- | ----------------------------------------- | --------- |
| Claude Code Docs                                | Task-first nav; shipped with releases; templated pages; runnable examples | Page templates + release-coupled updates  | Content   |
| Diátaxis (diataxis.fr)                          | tutorial / how-to / reference / explanation split                         | The organizing skeleton                   | Framework |
| Pydantic                                        | `mkdocstrings` + examples tested in CI                                    | Closest reference-generator cousin        | Tooling   |
| Polars                                          | User guide (how-to) + auto API; executed snippets                         | Guide-vs-reference split; adjacent domain | Content   |
| FastAPI                                         | Tutorial path; productive fast                                            | Onboarding learning track                 | Content   |
| NumPy                                           | Explicit Diátaxis restructure; peer audience                              | Structure maintained at scale             | Content   |
| Scientific Python (learn.scientific-python.org) | scikit-hep ecosystem conventions                                          | Native-feeling conventions                | Ecosystem |

Note: most PyData picks (Pydantic, FastAPI, …) are Material for MkDocs sites, so
they show where Zensical is heading; `zensical.org`'s own docs are the current
dogfood for what's possible today.

## Update workflow

Two tracks, by source of truth.

**Docstring track (scalable reference).** Docstrings (NumPy-style) are the
single source of truth for the `mkdocstrings`-generated reference; they are
tested via doctest. The reference's usefulness equals docstring quality, so
investing in docstrings improves the reference _and_ keeps it permanently in
sync — the lowest-effort, never-rots path, and it is already wired.

**Narrative track (Diátaxis guide pages).** Hand-written tutorials, how-tos, and
explanation. Author and maintain these with the persona-review loop proven on
the README: **rubric → personas → diverse drafts → parallel persona review →
synthesize** (accuracy beats style). Reuse the four personas from the README
review — New Adopter, PBT Practitioner, Domain Expert, Docs Editor — plus a
fifth for the docs: **Evaluator / Stakeholder**, who reads to assess and trust
the package but never runs code (an Awkward maintainer deciding to depend on it,
a reviewer, a citing author) and owns the "why it matters / is it reliable?"
story on the explanation and impact pages. Adapt per page. See
[`2026-06-17-readme-intro-persona-review.md`](./2026-06-17-readme-intro-persona-review.md).

**Update triggers by page type:**

- Reference (generated) → updates automatically with the code; no manual step.
- Stable narrative (tutorials, explanation, glossary) → review when the behavior
  they describe changes.
- Living pages (roadmap, bugs-found, FAQ, changelog) → explicit refresh trigger
  (e.g. bugs-found and roadmap reviewed each release).

**Anti-rot rules:**

- No duplicated prose across files (the README ↔ `docs/index.md` problem) — one
  source, link to it.
- Every example tested via doctest, so updates stay low-risk.

## Page plan (Diátaxis backlog)

`Q` = Diátaxis quadrant (Tut/How-to/Ref/Expl). Tags are first-pass and will
move.

| Page                                     | Q                  | Primary persona                  | Update trigger      | Status                              |
| ---------------------------------------- | ------------------ | -------------------------------- | ------------------- | ----------------------------------- |
| Overview (`index`)                       | Expl               | New Adopter                      | on intro change     | Exists (needs `docs/index.md` sync) |
| Installation                             | How-to             | New Adopter                      | on packaging change | Exists                              |
| Getting Started                          | Tut                | New Adopter                      | on API change       | Exists (thin)                       |
| API Reference                            | Ref                | All                              | auto (docstrings)   | Exists (generated)                  |
| Testing Awkward Array (incl. bugs found) | Expl + living log  | Domain Expert / New Adopter      | per release         | Proposed                            |
| Roadmap                                  | Expl/meta (living) | All                              | per milestone       | Proposed                            |
| How samples are generated & shrink       | Expl               | PBT Practitioner / Domain Expert | on algorithm change | Proposed                            |
| How-to cookbook (see below)              | How-to             | New Adopter / PBT Practitioner   | on API change       | Proposed                            |
| Glossary                                 | Ref/Expl           | All (bridges audiences)          | rare                | Proposed                            |
| Citation ("Cite this")                   | meta               | Scientific users                 | on release/DOI      | Proposed                            |
| Development methodology                  | Expl/contributing  | Contributor                      | rare                | Proposed                            |
| Audience bridges (2 primers)             | Expl               | New Adopter                      | rare                | Proposed                            |
| Supported dtypes/layouts table           | Ref                | Domain Expert                    | auto (from source)  | Proposed                            |
| FAQ / troubleshooting                    | How-to             | New Adopter                      | as issues arise     | Proposed                            |

**How-to cookbook recipes:** write a property-based test for code that takes
Awkward Arrays; constrain the generated arrays (`arrays()` options); reproduce &
debug a failing example; generate a specific shape/type; compose with other
Hypothesis strategies; use the lower-level strategies directly.

The maintainer's three requested pages are: _Testing Awkward Array (incl. bugs
found)_, _Roadmap_, and _How samples are generated & shrink_ — all explanation /
meta. The suggested adds deliberately fill the missing **how-to** layer plus
durable link targets (glossary, supported-types table).

## Open questions

- Long-term home for the living backlog (stay in this note vs its own `notes/`
  file).
- Add a `CITATION.cff` (and a Zenodo/DOI?) to back the citation page.
- Auto-generate the supported dtypes/layouts table from `SUPPORTED_DTYPE_NAMES`
  so it never rots.
- Whether to surface `CHANGELOG.md` inside the docs site.

## Split-later triggers

- Page-plan edited every release (a living backlog) → extract to its own
  `notes/` file.
- Exemplars section grows or more sites are added → promote to
  `.design/research/2026-06-17-docs-exemplars-research.md` (its proper bucket).
- Docs-initiative files exceed ~4 → promote to a `.design/docs/` subdirectory
  and add it to `.design/README.md`.

## Related

- [`2026-06-17-readme-intro-persona-review.md`](./2026-06-17-readme-intro-persona-review.md)
  — first worked example of the narrative-track workflow.
- [`.claude/rules/ci.md`](../../.claude/rules/ci.md) — docs build & deploy
  workflows (Zensical + `mike`).
- [`zensical.toml`](../../zensical.toml) — site config and the `mkdocstrings`
  bridge.
