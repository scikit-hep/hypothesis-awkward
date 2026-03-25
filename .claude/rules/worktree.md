# Worktree

- Use a worktree for tasks that don't need to modify the main branch directly.
  Create worktrees in `.claude/worktrees/` via the `EnterWorktree` tool.
- Name worktrees after the task or topic (e.g., `conda-forge`, `fix-typing`).
  If the topic isn't determined yet, omit the name to use the auto-generated one.
- After entering a worktree, work entirely from there. Use relative paths or
  the worktree path for all file operations.
- When work in a worktree is ready, confirm with the user before opening a PR
  from the worktree branch.
