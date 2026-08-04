---
paths:
  - ".claude/rules/persona-review-profile.md"
  - ".claude/rules/diataxis-declaration.md"
  - ".claude/personas/**"
---

# Reviewed-writer interface

These files are this repository's side of the `reviewed-writer` plugin's
interface; the plugin (pinned in `.claude/settings.json`) supplies the machinery
— the `write-doc` and `persona-review` skills and the `persona-reviewer` agent.
The profile's `##` headings are read by name by the plugin's skills; the persona
head files and the declaration file are read by its reviewers. Renaming or
restructuring any of them is an interface change, coordinated with the plugin
and with the counterpart repository (legendary-octo-happiness).
