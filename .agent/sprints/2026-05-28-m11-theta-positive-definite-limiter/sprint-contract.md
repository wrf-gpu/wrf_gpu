# Sprint Contract — M11: Dycore theta positive-definite limiter + guard accounting

**Sprint ID**: `2026-05-28-m11-theta-positive-definite-limiter`
**Worker**: codex gpt-5.5 xhigh
**Branch**: `worker/gpt/m11-theta-pd-limiter`
**Worktree**: `/tmp/wrf_gpu2_m11`
**Wall-time**: 6-18 h (target ≤ 1 day)
**GPU usage**: YES
**Sandbox**: `--sandbox danger-full-access`

## Why this sprint

M9.C confirmed theta divergence is a REAL model bug, not a comparator artifact. Theta mean RMSE 77 K across 24h is unphysical-large. The current `_limit_guarded_dynamics_state` (`src/gpuwrf/runtime/operational_mode.py:214-241`) clips theta to `[200K, 450K]` and falls back to a clipped *origin* state when out of envelope — a fail-closed guard that PRESERVES BOUNDED-FINITE RUNS but artificially suppresses diurnal evolution and masks dycore drift. M11 replaces this clip with a **positive-definite advection limiter** following WRF's approach (positivity-preserving with monotonicity, no mass loss). Critically, M11 also adds per-step **clip-count + first-clipped field/cell** logging (INV-10).

## Binding goal

A JAX-native GPU port of WRF v4 delivering Canary L2/L3 24-72 h RMSE on T2/U10/V10 **statistically equivalent** to CPU WRF v4 under **TOST** at predeclared margins on ≥30-case seasonal ensemble; ≥10× speedup preserved.

## Required inputs

1. `proofs/m9/divergence_map_v2.json` — confirms theta REAL_BUG
2. `src/gpuwrf/runtime/operational_mode.py` (theta guard, dycore update path)
3. `src/gpuwrf/dycore/**` — RK3 + acoustic substep + advection
4. `src/gpuwrf/coupling/physics_couplers.py` — theta-dependent physics
5. WRF source pattern: positive-definite advection from Skamarock 2008 or `dyn_em/module_advect_em.F` (only reference, no copy)

## Acceptance

### AC1 — Positive-definite limiter replaces clip

`src/gpuwrf/runtime/operational_mode.py`: `_limit_guarded_dynamics_state` is replaced or refactored so theta evolution uses a **mass-conserving positivity-preserving limiter** instead of a hard `[200K, 450K]` envelope clip. Specific design:
- After each dycore RK3 update step, theta increments are checked.
- If a cell's theta would go negative or exceed a physical maximum (~500K absolute), the increment is rescaled (not clipped) preserving total mass.
- The limiter is applied in the dycore (NOT post-physics — after-physics is a guard, this is INSIDE the dycore).

### AC2 — Clip-count logging (INV-10)

Each timestep records `theta_limited_cell_count` and `theta_first_limited_cell_xyz`. These flow into a `proofs/m11/limiter_diagnostics_24h.json` for a 24h Canary run.

### AC3 — 100-step parity preserved

`taskset -c 0-3 pytest -q tests/savepoint/test_dycore_100_steps.py` PASSES.

### AC4 — Skill non-regression vs M10 baseline

Re-run Canary 20260521 24h with new limiter. Re-run `scripts/m7_gpu_vs_cpu_skill_diff.py`. Emit `proofs/m11/post_m11_skill_diff.json`. Acceptance: **theta mean RMSE drops by ≥ 30 % vs `proofs/m9/divergence_map_v2.json` theta_mean_rmse, AND T2/U10/V10 RMSE does not get worse than `proofs/m10/post_m10_skill_diff.json`.**

### AC5 — Re-run operational trace with new theta evolution

`scripts/operational_trace_compare.py` re-runs on the post-M11 wrfouts. Emit `proofs/m11/divergence_map_v3.json` with same schema as v2. Compare theta + downstream fields (T2, PSFC, U/V/U10/V10) — they should improve if theta convention was indeed dycore-driven.

### AC6 — Worker report

`.agent/sprints/2026-05-28-m11-theta-positive-definite-limiter/worker-report.md`: standard format. Verdict `M11_COMPLETE` if AC1-AC5 all pass; `M11_PARTIAL` with explicit remaining gaps otherwise.

## Hard rules

1. **CPU pinning**: `taskset -c 0-3`.
2. **GPU usage**: YES — `--sandbox danger-full-access`.
3. **Files writable**: `src/gpuwrf/runtime/operational_mode.py`, `src/gpuwrf/dycore/**` (extend only), `proofs/m11/**`, `tests/savepoint/test_dycore_limiter.py` (NEW for clip-count verification), `.agent/sprints/2026-05-28-m11-theta-positive-definite-limiter/**`.
4. **Files NOT writable**: physics_couplers, BC, state contracts (except dycore-internal), governance.
5. **No remote push.**
6. **Manager repo ONLY**.
7. **Auto-notify on exit**: `tmux send-keys -t 0 "AGENT REPORT: m11 DONE exit=$?" Enter`.
8. **End with verdict**: `M11_COMPLETE` / `M11_PARTIAL` + headline theta RMSE reduction percentage.
