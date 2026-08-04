# Docs chapters

A living record of the pages that exist under `docs/`, one row per page. It
carries what a page needs but its own text cannot hold: the primary audience a
review briefs its personas with, and the trigger that says when the page is due
for a refresh.

"Primary audience" gives short forms of the persona names — who the page mainly
serves; all personas review every page.

Related: the marker rules in
[`.claude/rules/diataxis-review.md`](../../.claude/rules/diataxis-review.md),
and the page backlog in
[`2026-06-17-02-Docs-plan.md`](../notes/2026-06-17-02-Docs-plan.md).

| Page                                        | Primary audience                         | Update trigger      |
| ------------------------------------------- | ---------------------------------------- | ------------------- |
| `index.md`                                  | Researcher / Evaluator                   | on intro change     |
| `guide/index.md`                            | — (nav index)                            | on page added       |
| `guide/installation.md`                     | Researcher / Evaluator                   | on packaging change |
| `guide/getting-started.md`                  | Downstream dev / Researcher              | on API change       |
| `guide/testing-awkward-array.md`            | Awkward core dev / Evaluator             | per release         |
| `guide/roadmap.md`                          | Awkward core dev / Downstream dev        | per milestone       |
| `guide/generating-and-shrinking-samples.md` | Hypothesis-PBT expert / Awkward core dev | on algorithm change |
| `reference/index.md`                        | All                                      | on module added     |
| `reference/util.md`                         | All                                      | auto (docstrings)   |
| `reference/strategies/builtins.md`          | All                                      | auto (docstrings)   |
| `reference/strategies/constructors.md`      | All                                      | auto (docstrings)   |
| `reference/strategies/contents.md`          | All                                      | auto (docstrings)   |
| `reference/strategies/numpy.md`             | All                                      | auto (docstrings)   |
