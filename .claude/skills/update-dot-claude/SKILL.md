---
name: update-dot-claude
description: Audit and update `.claude/` agent docs against the repo
disable-model-invocation: true
---

Review the documentation in `.claude/` against the current state of the
repository. Update any docs that are outdated. These are the agent's own
instructions, so the goal is accuracy — edit them to match reality.

## Steps

1. **Explore the repo** — Use Glob and Read to understand the current source
   layout (`src/`), tests (`tests/`), and config files (`pyproject.toml`,
   `.github/`).

2. **Audit `.claude/`** — Read every file under `.claude/` (except `settings*`).
   For each file, check whether its content still accurately reflects the repo:

   - `CLAUDE.md` — project overview, build commands, code architecture, strategy
     listing, usage conventions, code style rules.
   - `rules/ci.md` — CI matrix, dependency files, release workflows (path-scoped
     to `.github/**` and `pyproject.toml`).
   - `rules/testing-patterns.md` — test patterns, example code (path-scoped to
     `tests/**`).
   - `rules/docs.md` — docs conventions: build, structure, examples (path-scoped
     to `docs/**`).
   - `rules/docs-voice.md` — voice rules for user-facing prose (path-scoped to
     `README.md` and `docs/**`).
   - `rules/diataxis-review.md` — Diátaxis review guidance for the
     `docs-persona-*` agents (always loaded so subagents receive it).
   - `agents/` and `skills/` — check that the file paths and repo facts they
     reference still exist.

   Pick up any rule files added later.

3. **Optimize context** — Review always-loaded context (files without `paths:`
   frontmatter) for size and relevance:

   - Check total line counts. Flag any rule file over ~50 lines that isn't
     path-scoped.
   - Check whether any section in `CLAUDE.md` contains details that Claude could
     discover from code (e.g., per-function listings). Suggest trimming.
   - Check whether any rule file should be path-scoped to reduce default context
     load.

4. **Report findings** — Summarize what is outdated and what changes you
   propose. Group by file.

5. **Apply updates** — Edit the outdated docs to match reality. Keep the
   existing voice and style. Do not rewrite content that is still accurate.

## Guidelines

- Do not touch `settings.json` or `settings.local.json`.
- Prefer minimal, targeted edits over wholesale rewrites.
