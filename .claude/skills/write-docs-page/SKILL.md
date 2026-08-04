---
name: write-docs-page
description:
  Author or substantially revise a docs page via the persona-review workflow
disable-model-invocation: false
---

Invoke the `reviewed-writer:write-doc` skill: it reads this repository's profile
(`.claude/rules/persona-review-profile.md`), whose Document section scopes the
run to one page under `docs/`, and runs the authoring workflow — rubric, three
diverse drafts, persona panel, fact-check, synthesis, and re-review until every
persona ships. Pass along the page being authored or revised, the change driving
it, and any scoping the user supplied.
