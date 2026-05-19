# Memory Patch Proposal

## Scope

One auto-memory patch: `git worktree add` is a viable isolation pattern for sub-agent dispatches that solves the manager-worker shared-worktree contamination problem.

## Evidence

Codex reviewer attempt 2 used `git worktree add /home/enric/src/wrf_gpu2-review-m2-triton/` to isolate its workspace, committed cleanly there, pushed the branch tip back. The reviewer report committed as `8157694` is on the reviewer branch and accessible from the main worktree without any merge conflicts or contamination.

## Proposed Destination

Append to `/home/enric/.claude/projects/-home-enric-src-wrf-gpu2/memory/feedback_manager_autonomy.md` "Operational addition 2026-05-19 (post M2-S2)" section.

## Patch

Append to the existing manager-during-worker section:

```markdown
## Update 2026-05-19 (post M2-S6): git worktree as the cleaner isolation pattern

Codex (when given sufficient autonomy) can use `git worktree add` to isolate its workspace. Solved the contamination problem perfectly on M2-S6 reviewer attempt 2. Pattern for dispatch_role.sh to adopt:

1. Before launching the agent: `git worktree add /home/enric/src/wrf_gpu2-<role>-<sprint> <role-branch>`.
2. Run the agent with `-C` pointing at that worktree.
3. Agent commits + pushes its branch.
4. Manager fetches branch tip into main worktree (`git checkout <role-branch>` works once worktree is removed).
5. After deliverable captured: `git worktree remove --force` the agent's worktree.

Deferred refactor: dispatch_role.sh integration. For now, manual recovery is well-understood.
```

## Reviewer Status

Reviewer Status: not required — process observation, not a rule change.
