---
name: wrf-gpu-worker
description: Use when implementing a bounded WRF-GPU change.
---

# Implementation worker

Read `AGENTS.md` and `docs/agent-workflow/manager/SKILL.md` from the repository
root. Accept a contract naming the input revision, scope, acceptance tests,
owner and stop conditions. Ask the manager to resolve missing authority.

- Verify the branch/revision and worktree status before editing. Use an isolated
  worktree or clone; do not overwrite another worker's changes.
- Reproduce the failure and add a regression test where appropriate. Make the
  smallest fix; keep unrelated refactors, generated output and local state out.
- Preserve WRF numerical/output semantics and frozen tolerances. No performance
  shortcut may bypass scientific gates. Record any intended semantic change.
- Run affected checks and retain exact commands, exits, failures and skips. Do
  not report intended execution as completed execution.
- Seal the candidate commit for testing/review. Return changed files, rationale,
  reproduction before/after, evidence and unresolved risks to the manager.
- Do not independently publish releases, start paid hardware, modify tags or
  seize integration ownership. Stop on scope conflicts or missing oracle data.
