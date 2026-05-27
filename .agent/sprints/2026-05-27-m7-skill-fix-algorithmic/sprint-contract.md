# Sprint Contract — M7 Skill Fix (Algorithmic — Sprints A+B+C combined)

**Sprint ID**: `2026-05-27-m7-skill-fix-algorithmic`
**Created**: 2026-05-27 (user direction: finish project + publication)
**Status**: READY — top priority
**Predecessors**:
- `.agent/sprints/2026-05-27-m7-skill-regression-rca-opus/top_3_suspects.md` (Sprint A+B+C+D+E sequence)
- `.agent/sprints/2026-05-27-m7-skill-regression-rca-codex/worker-report.md` (MULTIPLE_CONTRIBUTORS — confirms)
- `.agent/decisions/MILESTONE-M7-CLOSEOUT-AMENDMENT.md` (M7-PENDING-SKILL-FIX)

## Objective

Apply the three algorithmic fixes the RCA sprints converged on, in one combined sprint. The opus tester explicitly recommended A+B together (same RK step composition) plus C. This sprint scopes A+B+C; the LU_INDEX audit (Sprint D) runs as a parallel opus sprint.

**Sprint A — Remove the dycore theta/mu state reset.** In `src/gpuwrf/runtime/operational_mode.py:_physics_boundary_step` lines 548-563 (the `disable_guards=False` branch), the post-RK state for `theta`, `mu`, `mu_total`, `mu_perturbation` is currently overwritten with the pre-step values, discarding the RK3 + acoustic advance. Replace with: keep the RK advance; add an **inline positive-definite limiter** for `theta` (clamp to `[200K, 400K]` for lower 30 levels, `[250K, 700K]` for upper 14 levels, matching the existing `theta_bounds` schema in `feedback_validation_philosophy.md`) and a positive-definite limiter for `mu_perturbation` so the original motivation for the guard branch (preventing nonfinite) is preserved without throwing the dynamics away.

**Sprint B — Wire surface→PBL coupling.** In `src/gpuwrf/coupling/physics_couplers.py:mynn_adapter` (lines 250-281), the surface heat/moisture/momentum-flux bottom BC inputs are currently `jnp.zeros_like(theta_columns)`. Reorder `_physics_boundary_step` so `surface_adapter` runs **before** `mynn_adapter`, then pass `state.theta_flux`, `state.qv_flux`, and a tau-derived momentum-flux pair into `MynnPBLColumnState` instead of zeros. Verify units/sign convention matches `physics/surface_layer.py` outputs.

**Sprint C — Enable RRTMG at WRF default cadence.** Change `DailyPipelineConfig.radiation_cadence_steps` default from `999999` to `180` (= 30 min × 60 s / dt=10 s). Verify `rrtmg_adapter` actually applies heating-rate tendencies to theta when invoked (read `coupling/physics_couplers.py:315-365`).

## Acceptance

- **AC1 — A applied**: state reset code removed; inline limiters present in the RK3 update. Verify by reading the modified `_physics_boundary_step`.
- **AC2 — Hour-to-hour theta variation**: re-run the 24h pipeline on 20260521. `theta_lower_30_max_k` must vary between hourly snapshots by **at least 5 K** (compared to the current behavior of identical values across all 24 snapshots).
- **AC3 — B applied**: `mynn_adapter` receives non-zero surface flux inputs; ordering in `_physics_boundary_step` is `surface_adapter` before `mynn_adapter`. Add a unit test that confirms a synthetic column with `theta_flux = +200 W/m²` produces a measurable theta increase between pre-surface and post-PBL state.
- **AC4 — C applied**: `radiation_cadence_steps=180` is the default. The 24h pipeline invokes `rrtmg_adapter` 8640/180 = 48 times.
- **AC5 — Bounds preserved**: post-fix 24h pipeline produces all-finite output; theta bounds within `[200K, 700K]`; |U|,|V| ≤ 100; |W| ≤ 50; per `feedback_validation_philosophy.md` envelope.
- **AC6 — Skill re-measurement**: re-run `scripts/m7_gpu_vs_cpu_skill_diff.py` on the post-fix 20260521 GPU forecast vs CPU reference. Emit `.agent/sprints/2026-05-27-m7-skill-fix-algorithmic/post_fix_skill_diff.json` with: GPU T2/U10/V10 BIAS/RMSE/MAE, CPU baselines, relative deltas, verdict `±20% tolerance met` / `improved but partial` / `no improvement`.
- **AC7 — Honest speedup re-check**: re-run `scripts/m7_cpu_per_domain_timing.py` against the same Gen2 reference; produce `post_fix_speedup.json` with the apples-to-apples ratio after fix (expected to dip slightly because radiation now runs 48× per 24h).
- **AC8 — Regression preservation (HARD)**: 20260521 multi-step parity step 2 = 0.0 bitwise; B6 savepoint parity 0.0 bitwise; D2H inter-kernel = 0; restart bitwise PASS. Sprint MUST NOT regress these invariants.
- **AC9 — Tests**: add `tests/test_m7_skill_fix_algorithmic.py` covering: (a) theta/mu update flows through RK3 to next-step state, (b) `mynn_adapter` consumes non-zero surface flux inputs, (c) limiter activates on synthetic out-of-bounds input, (d) `radiation_cadence_steps=180` default.
- **AC10 — Worker report**: verdict `SKILL_FIXED` / `SKILL_IMPROVED_PARTIAL` / `BLOCKED`.

## Files Worker May Modify

- `src/gpuwrf/runtime/operational_mode.py` (Sprint A: the guard branch lines 548-563; preserve surrounding code)
- `src/gpuwrf/coupling/physics_couplers.py` (Sprint B: mynn_adapter + surface_adapter wiring; Sprint A inline limiters if they live here)
- `src/gpuwrf/integration/daily_pipeline.py` (Sprint C: `radiation_cadence_steps` default = 180)
- `tests/test_m7_skill_fix_algorithmic.py` (NEW)
- `.agent/sprints/2026-05-27-m7-skill-fix-algorithmic/**`

## Files Worker Must Not Modify

- `src/gpuwrf/dynamics/**` — the dycore is correct per B6; the guard above it is the bug
- `src/gpuwrf/contracts/state.py`, `contracts/precision.py` — State schema unchanged
- `src/gpuwrf/io/**` — except this sprint does NOT touch land_state.py (Sprint D handles LU_INDEX)
- `src/gpuwrf/validation/**` — frozen
- `src/gpuwrf/runtime/checkpoint.py` — frozen post-M7
- governance files
- `/mnt/data/canairy_meteo/**`

## Hard Rules

1. **Preserve invariants** (AC8). 20260521 0.0 bitwise + B6 + D2H + restart = hard gates. If any regress, the fix must be revised before commit.
2. **No re-enabling `disable_guards=True`**. Production runs always with `disable_guards=False`. The fix replaces the reset semantics, not the flag's purpose.
3. **CPU pinning**: `taskset -c 0-3` for any orchestrator.
4. **GPU**: yes, runs forecasts. Coexists with publication draft worker (CPU-only) — no contention.
5. **No remote push.** Local commit on `worker/gpt/m7-skill-fix-algorithmic` only.
6. **Honest BLOCKED**: if AC6 shows no improvement, dispatch decision goes back to the manager; don't fudge the verdict.
7. **Stay in scope**: A + B + C only. Do NOT touch LU_INDEX (Sprint D), lateral boundary width (Sprint E), or land state refresh (Sprint D follow-up) in this sprint.

## Proof Objects

- `.agent/sprints/2026-05-27-m7-skill-fix-algorithmic/post_fix_skill_diff.json` (AC6 — the gate)
- `.agent/sprints/2026-05-27-m7-skill-fix-algorithmic/post_fix_speedup.json` (AC7)
- `.agent/sprints/2026-05-27-m7-skill-fix-algorithmic/post_fix_bounds.json` (AC5)
- `.agent/sprints/2026-05-27-m7-skill-fix-algorithmic/invariant_preservation.json` (AC8 — 20260521 + B6 + D2H + restart all PASS)
- `.agent/sprints/2026-05-27-m7-skill-fix-algorithmic/worker-report.md` (AC10)
- `tests/test_m7_skill_fix_algorithmic.py` (AC9)

## Dispatch

- Worker: codex gpt-5.5 xhigh
- Wall-time: 6-12 h (substantial — three coupled fixes + re-validation)
- Branch: `worker/gpt/m7-skill-fix-algorithmic`
- Worktree: `/tmp/wrf_gpu2_skillfix`
- GPU usage: YES — runs 24h forecasts for AC2/AC6 re-validation
- Parallel-safe: publication draft (CPU) + LU_INDEX audit (opus CPU) both have zero GPU contention
