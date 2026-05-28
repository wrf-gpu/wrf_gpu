# Sprint Contract — Project-Reset Blinded Planner (codex GPT-5.5 xhigh)

**Sprint ID**: `2026-05-28-project-reset-blinded`
**Role**: BLINDED PLANNER — you do NOT see the manager's draft plan
**Worker**: codex gpt-5.5 xhigh
**Branch**: `worker/gpt/project-reset-blinded`
**Worktree**: `/tmp/wrf_gpu2_pr_blinded`
**Wall-time**: 60-90 min
**GPU usage**: NONE — pure analysis sprint

## Binding goal

> A JAX-native GPU port of WRF v4 that delivers Canary L2/L3 forecasts whose 24-72 h RMSE on T2, U10, V10 is not statistically distinguishable from CPU WRF v4 (paired t-test p > 0.05 on a ≥ 15-case ensemble across seasons), while preserving ≥ 10× speedup vs 28-rank CPU WRF on the same workstation.

## Objective

Build a project-reset plan FROM SCRATCH that takes the project from its current state to the binding goal. **DO NOT READ** `.agent/decisions/PROJECT-RESET-PLAN-DRAFT.md`. **DO NOT READ** `.agent/sprints/2026-05-28-project-reset-critic/`. Your job is to be the independent second opinion — the manager's plan will be compared against yours and merged.

## Permitted inputs (read in order)

1. `PROJECT_CONSTITUTION.md`, `AGENTS.md`
2. `.agent/decisions/ADR-027*.md` — D2H-zero invariant
3. `.agent/decisions/MILESTONE-M7-CLOSEOUT-AMENDMENT.md` — 156→22.26× correction context
4. `.agent/sprints/2026-05-27-m7-skill-fix-iter2/post_iter2_skill_diff.json` — last skill measurement
5. `.agent/sprints/2026-05-27-m7-skill-regression-rca-opus/top_3_suspects.md` — last RCA
6. `src/gpuwrf/operational_mode.py`, `src/gpuwrf/coupling/physics_couplers.py`, `src/gpuwrf/state/state.py`
7. `tests/savepoint/` — what B6 actually tests
8. `scripts/run_canary_*.sh` — operational entry points
9. `proofs/` — any standing proof artifacts

## FORBIDDEN inputs (do not read these — they would contaminate your independence)

- `.agent/decisions/PROJECT-RESET-PLAN-DRAFT.md`
- `.agent/sprints/2026-05-28-project-reset-critic/**`
- Anything authored by the manager today (2026-05-28) under `.agent/decisions/`

## Acceptance

Produce `.agent/sprints/2026-05-28-project-reset-blinded/plan.md` answering:

- **B1 — Position assessment**: where is the project now, weighted by importance to the binding goal? Produce a table with at least 6 weighted blocks summing to 100 %. Estimate overall completion %.

- **B2 — Milestone roadmap**: list every milestone needed from current state to binding goal. For each: definable numeric proof, estimated weeks, estimated Δ% completion gained, risk Low/Medium/High.

- **B3 — Dependency graph**: which milestones genuinely block which? Where can work parallelise?

- **B4 — Critical path**: what's the single longest chain in your dependency graph? What's the shortest plausible time to closure if that chain runs perfectly?

- **B5 — Invariant ladder**: list the invariants that every sprint close must satisfy to "constantly improve without breaking." Be specific about measurement.

- **B6 — Multi-AI verification pattern**: who reviews what. Mandatory cross-checks. Where Gemini-class oversight is justified vs over-budget.

- **B7 — Risk register**: top 5 risks. For each: impact, likelihood, mitigation.

- **B8 — What can fail silently**: list 3 ways this project could deliver milestone closures while still missing the binding goal. (E.g. overfitting to the pinned 5-day case; statistical-equivalence test with low statistical power; performance ratchet that masks correctness regression.)

- **B9 — Sprint sizing recommendation**: how long should each sprint be? How many sprints per milestone? What does a "good" sprint contract look like — write one example for the first milestone in your roadmap.

- **B10 — One paragraph for the principal**: write the single paragraph you would say to the principal to summarise the plan. Plain language, no jargon. Honest about uncertainty.

## Hard rules

1. **Independence is the point.** If you find yourself second-guessing what the manager wrote — STOP. You haven't read it; don't reverse-engineer it.
2. **Numbers cited to files.** No vague estimates.
3. **CPU pinning**: `taskset -c 0-3`.
4. **No GPU runtime.** Reading + writing only.
5. **No code changes.** Planning sprint only.
6. **No remote push.** Local commit on `worker/gpt/project-reset-blinded`.
7. **Manager repo ONLY** — do not touch `/home/enric/src/wrf_gpu/`.
8. **End with verdict**: `PLAN_COMPLETE / PLAN_PARTIAL` and one-line summary.

## Files you may modify

- `.agent/sprints/2026-05-28-project-reset-blinded/**`

## Files you must not modify

- Everything else.

## Dispatch

Spawn me. Read inputs (NOT the forbidden ones). Write plan. Commit. Exit.
