---
name: write-docs-page
description:
  Author or substantially revise a docs page via the persona-review workflow
disable-model-invocation: false
---

Invoke the `reviewed-writer:write-doc` skill. It drives the
`reviewed-writer:diataxis-writer` and `reviewed-writer:diataxis-persona-panel`
agents, which read `.claude/rules/persona-review-profile.md`, whose Document
section scopes the run to one page under `docs/`. The writer runs the authoring
workflow — rubric, diverse drafts, fact-check, synthesis, revision — the panel
runs a persona round on each text, and `write-doc` iterates until every persona
ships and relays the outcomes. Pass along the page being authored or revised,
the change driving it, and any scoping the user supplied.
