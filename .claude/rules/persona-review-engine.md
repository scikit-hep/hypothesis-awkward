---
paths:
  - ".claude/skills/write-docs-page/SKILL.md"
  - ".claude/rules/persona-review-profile.md"
  - ".claude/rules/diataxis-review.md"
  - ".claude/agents/docs-persona-*.md"
---

# Persona-review engine

The body of `.claude/skills/write-docs-page/SKILL.md` is repository-agnostic, to
be eventually extracted as a plugin: every repository-specific value lives in
`.claude/rules/persona-review-profile.md`, whose `##` headings are the interface
the engine reads by name. `.claude/rules/diataxis-review.md` is likewise
repository-agnostic and plugin-bound, mirrored whole-file; its
repository-specific counterpart is `.claude/rules/diataxis-declaration.md`.
Below their persona-specific heads, the `docs-persona-*` agent files share a
byte-identical machinery tail, starting at the "**Review by quadrant.**" line
and mirrored across both repositories' persona files.
