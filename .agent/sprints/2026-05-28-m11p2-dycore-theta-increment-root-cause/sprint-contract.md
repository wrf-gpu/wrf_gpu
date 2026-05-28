# Sprint Contract — M11.2: Dycore Theta-Increment Root Cause

**Sprint ID**: `2026-05-28-m11p2-dycore-theta-increment-root-cause`
**Worker**: codex gpt-5.5 xhigh
**Branch**: `worker/gpt/m11p2-dycore-theta-increment-rc`
**Worktree**: `/tmp/wrf_gpu2_m11p2`
**Wall-time**: 6-18 h (target ≤ 1 day)
**GPU usage**: YES
**Sandbox**: `--sandbox danger-full-access`

## Why this sprint

M17 1h diagnostic harness found: **`theta_in_bounds` first violates at step 141 with `first_violation_operator = dycore_rk3`.** This is the deeper bug that M11's positive-definite limiter is currently masking. M11 limiter clips every single one of 8,640 timesteps (up to 315k cells/step), with mass residual 0.027K. That's the limiter doing all the work the dycore should do correctly. M11.1 fixed p/ph silent flatlines but did not touch the theta increment path. M11.2 fixes the root cause.

If M11.2 succeeds, the limiter clip count should drop dramatically (target: <10% of steps need any clipping; mass residual unchanged), and station T2 RMSE should recover from the M11 regression (10.80 → 13.11K should return ≤10.80K).

## Binding goal

A JAX-native GPU port of WRF v4 delivering Canary L2/L3 24-72 h RMSE on T2/U10/V10 **statistically equivalent** to CPU WRF v4 under **TOST** at predeclared margins on ≥30-case seasonal ensemble; ≥10× speedup preserved.

## Required inputs

1. `proofs/m17/diagnostic_report_after_fix.json` — has `theta_in_bounds` first violation at step 141
2. `proofs/m11/limiter_diagnostics_24h.json` — 8640/8640 steps clipped, max 315k cells, mass residual 0.027
3. `src/gpuwrf/dynamics/**` — dycore source (theta update path)
4. `src/gpuwrf/runtime/operational_mode.py` — limiter call site (post-M11)
5. WRF reference: `dyn_em/module_em.F` rk_scalar_tend, advect_scalar_pd, and theta tendency assembly

## Acceptance

### AC1 — Root cause identified

`.agent/sprints/2026-05-28-m11p2-dycore-theta-increment-root-cause/root_cause_analysis.md`:
- Identify the exact line(s) in `src/gpuwrf/dynamics/**` that produce the bad theta increments
- Diagnose: missing operator, wrong sign, wrong coefficient, missing density factor, missing advection-pd guard, vertical-coordinate misalignment, or other
- Cite WRF reference for the correct formulation

### AC2 — Fix applied

Minimal diff in `src/gpuwrf/dynamics/**`. Document why each line changed.

### AC3 — Harness re-verification (PRE+POST per critic gate)

Re-run `scripts/run_diagnostic_harness.py --hours 1 --radiation-cadence-steps 999999`. Emit `proofs/m11p2/diagnostic_report_after_fix.json`. Acceptance:
- `theta_in_bounds` no longer violates within 1h horizon, OR first-violation step pushed to >500 (significant improvement)
- `dycore_rk3` remains `ACTIVE`
- No other operator regresses

### AC4 — Limiter activity dropped

Re-run `proofs/m11p2/limiter_diagnostics_24h.json`. Target:
- Limited step count: ≤ 10% of total steps (from 100%)
- Max cells limited per step: ≤ 10% of M11's 315k (from 100%)
- Mass residual: ≤ 0.027 (no regression)

### AC5 — 100-step parity preserved

`taskset -c 0-3 pytest -q tests/savepoint/test_dycore_100_steps.py` PASSES.

### AC6 — 24h skill recovery

Re-run Canary 20260521 24h. Emit `proofs/m11p2/post_m11p2_skill_diff.json`. Acceptance:
- T2 RMSE recovers ≤ M10 baseline (10.80 K); ideally drops vs M10
- Speedup ≥ 14× (no severe regression vs M11's 20.60×)

### AC7 — Worker report

`.agent/sprints/2026-05-28-m11p2-dycore-theta-increment-root-cause/worker-report.md`. Verdict `M11P2_COMPLETE` if AC1-AC6 all pass; `M11P2_PARTIAL` with explicit unfinished gaps otherwise.

## Hard rules

1. **CPU pinning**: `taskset -c 0-3`.
2. **GPU usage**: YES — `--sandbox danger-full-access`. Coordinate with parallel RRTMG-triage + M14 workers.
3. **Files writable**: `src/gpuwrf/dynamics/**`, `proofs/m11p2/**`, `.agent/sprints/2026-05-28-m11p2-dycore-theta-increment-root-cause/**`.
4. **Files NOT writable**: physics couplers, operational_mode (only acceptable: removing/relaxing the limiter if AC4 is achieved — document why), BC, state contracts except dycore-internal, governance.
5. **No remote push.**
6. **Manager repo ONLY**.
7. **Auto-notify on exit**: `tmux send-keys -t 0 "AGENT REPORT: m11p2 DONE exit=$?" Enter`.
8. **End with verdict**: `M11P2_COMPLETE` / `M11P2_PARTIAL` + headline limiter-activity drop percentage + T2 RMSE post-fix.
