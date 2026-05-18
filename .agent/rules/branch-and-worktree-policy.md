# Branch And Worktree Policy

- Use purpose-named branches: `worker/gpt/<sprint>`, `reviewer/opus/<sprint>`, `tester/sonnet/<sprint>`, `manager/integration/<sprint>`.
- One owner per core file during a sprint.
- Parallel work requires frozen interfaces and non-overlapping file ownership.
- Do not rebase or force-push another agent's branch.
- Integration happens only after reports and proof objects exist.
