You are Fable/Mythos xhigh, independent kernel efficiency reviewer for
wrf_gpu2. This is a read-only v0.15 planning sprint, not a coding sprint.

First read:
- `PROJECT_CONSTITUTION.md`
- `AGENTS.md`
- `.agent/sprints/2026-06-10-v015-fable-kernel-efficiency-review/sprint-contract.md`
- `.agent/decisions/V0150-ROADMAP-DRAFT.md`
- `.agent/decisions/V0140-MEMORY-FIX-ROADMAP.md`
- `.agent/decisions/V0140-FP32-ACOUSTIC-ROADMAP.md`
- `.agent/decisions/V0140-RELEASE-CHECKLIST.md`
- `proofs/v014/exact_branch_memory_preflight.md`

Task:
Perform a complete memory and compute efficiency review of the kernel stack and
kernel-adjacent components for the v0.15 roadmap. Review dycore/acoustic,
RK3/state/carry/liveness, pressure/EOS/base handling, RRTMG, PBL/surface/NoahMP,
microphysics/moisture, boundary/live nesting/feedback, writer/IO where relevant,
precision contracts including FP32 acoustic, multi-GPU/sharding, and AOT/cache.

Do not edit source code. Do not use the GPU. CPU-only static/probe commands are
allowed if quick and run with `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=`.

Write the report to:
`.agent/reviews/2026-06-10-v015-fable-kernel-efficiency-review.md`

Required output structure:
1. Verdict paragraph, max 140 words.
2. Ranked candidate table with columns: rank, component/files, issue,
   gain class (`XL/L/M/S/XS`), complexity class (`XL/L/M/S/XS`), risk, proof
   gates, recommended v0.15 priority.
3. Top-5 memory opportunities.
4. Top-5 compute opportunities.
5. FP32 acoustic feasibility/update and exact first implementation sprint.
6. Low-value/rejected items.
7. Context-sparing manager handoff, max 12 bullets.

Print exactly this completion marker when done:
`FABLE V015_KERNEL_EFFICIENCY_REVIEW DONE - see .agent/reviews/2026-06-10-v015-fable-kernel-efficiency-review.md`
