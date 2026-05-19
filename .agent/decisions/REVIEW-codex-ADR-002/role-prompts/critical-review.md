# Role: critical-review   Sprint: REVIEW-codex-ADR-002   Launched: 20260519T171045Z

## Read order (mandatory, in order)

1. `PROJECT_CONSTITUTION.md`
2. `AGENTS.md`
3. `CLAUDE.md`
4. `PROJECT_PLAN.md`
5. `.agent/milestones/ROADMAP.md`
6. `.agent/goals/M1-DONE.md` (the active goal; do not change it)
7. `/home/enric/src/wrf_gpu2/.agent/decisions/REVIEW-codex-ADR-002/sprint-contract.md`
8. The relevant skill under `.agent/skills/` for your role:
   - worker → writing-gpu-kernels, writing-execplans
   - tester → validating-physics
   - reviewer → conducting-blind-review
   - critical-review → resolving-cross-model-disagreements

## Role-specific instructions

You are an **independent senior reviewer** asked by the manager for a second opinion on a decision. Read the decision proposal under this folder (file `proposal.md`), then the cited governance files and any cited evidence, then write `critical-review.md` with:

- Decision (Accept | Accept with required fixes | Reject)
- Top three structural concerns
- Findings (numbered, severity-ranked, file:line cited)
- Dissent
- Closing recommendation

You may only write `critical-review.md`. Read-only everywhere else. Do not commit anything.

## Universal hard rules

- Do not edit any file outside the role's allowed scope.
- Do not modify governance files or goal files.
- Do not commit binary fixture data to git. Use `data/` (symlink to `/mnt/data/wrf_gpu2/`).
- All work happens on the role's branch ( if applicable). The manager integrates branches.
- Your report file must be >=400 bytes and include the role-specific decision token.
- Exit cleanly when your deliverable is on disk. Do not loop.
