---
paths:
  - ".claude/skills/write-docs-page/SKILL.md"
  - ".claude/rules/persona-review-profile.md"
  - ".claude/skills/persona-review/**"
  - ".claude/agents/persona-reviewer.md"
---

# Persona-review engine

The body of `.claude/skills/write-docs-page/SKILL.md` is repository-agnostic, to
be eventually extracted as a plugin: every repository-specific value lives in
`.claude/rules/persona-review-profile.md`, whose `##` headings are the interface
the engine reads by name. The `persona-review` skill — its whole directory,
including the Diátaxis core at `references/diataxis-review.md` — and the
`persona-reviewer` agent are likewise repository-agnostic and plugin-bound,
mirrored whole-file across both repositories. Their repository-specific
counterparts are the profile, `.claude/rules/diataxis-declaration.md`, and the
persona head files in `.claude/personas/`.
