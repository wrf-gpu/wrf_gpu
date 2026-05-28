# Sprint Contract — M11.3: Coordinated Dycore Fix (Restore Advection + mu Total + theta_1 Decouple)

**Sprint ID**: `2026-05-28-m11p3-coordinated-dycore-fix`
**Worker**: codex gpt-5.5 xhigh
**Branch**: `worker/gpt/m11p3-coordinated-dycore-fix`
**Worktree**: `/tmp/wrf_gpu2_m11p3`
**Wall-time**: 6-18 h (target ≤ 1 day)
**GPU usage**: YES
**Sandbox**: `--sandbox danger-full-access`
**Per Gemini agy review** (`.agent/sprints/2026-05-28-agy-dycore-deep-review/findings.md`)

## Why this sprint

agy proved that THREE coordinated dycore bugs are interacting to produce the operational divergence. Single-line fixes regress because each candidate cannot compensate for the other two bugs. This sprint applies all 3 fixes together in a single coordinated change, then verifies with the harness + 24h pipeline.

agy ALSO proved the existing `test_dycore_100_steps.py` is a JAX-vs-JAX self-compare tautology (the comparator reads back what it just wrote). The 100-step test will continue to pass regardless of correctness. That test gap is **out of scope** for M11.3 (separate test-comparator-oracle-rewrite sprint will fix it). M11.3 focuses on the model-code fix.

## Binding goal

A JAX-native GPU port of WRF v4 delivering Canary L2/L3 24-72 h RMSE on T2/U10/V10 **statistically equivalent** to CPU WRF v4 under **TOST** at predeclared margins on ≥30-case seasonal ensemble; ≥10× speedup preserved.

## Required inputs (read in order)

1. `.agent/sprints/2026-05-28-agy-dycore-deep-review/findings.md` — primary directive
2. `proofs/m11p2/diagnostic_report_after_fix.json` — pre-fix state
3. `proofs/m11p2/limiter_diagnostics_24h.json` — limiter saturation evidence
4. `src/gpuwrf/runtime/operational_mode.py` — `_rk_scan_step` lines ~588-617
5. `src/gpuwrf/dynamics/core/acoustic.py` — `_decouple_theta_after_advance` (~line 185-200) + `acoustic_substep_core` return (~line 255-275)
6. `src/gpuwrf/dynamics/` — anything called `compute_advection_tendencies` or `advect_*` that exists but isn't being called from operational mode
7. `tests/savepoint/test_dycore_100_steps.py` (don't touch — it will pass regardless because comparator is tautological; document that it passes)

## Acceptance — apply ALL THREE fixes coordinated

### AC1 — FIX 1: Restore advection in `_rk_scan_step`

In `src/gpuwrf/runtime/operational_mode.py`, modify `_rk_scan_step`'s `advance_stage` inner function to also call advection tendencies + combine them with the horizontal pressure gradient tendencies BEFORE the acoustic substep. The advection function should already exist somewhere under `src/gpuwrf/dynamics/`; if it does, import + call it. If it does not exist (was deleted), reimplement minimal WRF-faithful advection for U/V/W/theta/QVAPOR.

Document the function called and the WRF reference at `dyn_em/module_em.F:rk_scalar_tend` or equivalent.

### AC2 — FIX 2: `acoustic.py` mu=advanced["mu"] not mu_delta

In `src/gpuwrf/dynamics/core/acoustic.py:255-275` (function `acoustic_substep_core`), replace `state.replace(mu=mu_delta, ...)` with `state.replace(mu=advanced["mu"], ...)`. One-line change.

### AC3 — FIX 3: `acoustic.py` theta_1 in _decouple_theta_after_advance

In `src/gpuwrf/dynamics/core/acoustic.py:185-195`, replace `state.theta` with `state.theta_1` in the numerator of `_decouple_theta_after_advance`. One-line change.

### AC4 — Harness post-fix verification (1h, radiation off)

Run `scripts/run_diagnostic_harness.py --hours 1 --radiation-cadence-steps 999999`. Emit `proofs/m11p3/diagnostic_report_after_fix.json`. Acceptance:
- `dycore_rk3` verdict `ACTIVE`
- `wind_in_bounds` NO violation within 1h (M11.2 worktree showed step 72)
- `theta_in_bounds` NO violation within 1h (M11.2 showed step 85)
- First nonfinite NEVER (M11.2 showed step 93)
- No regression on other operators

### AC5 — Limiter activity drops dramatically

Run the 24h limiter diagnostic. Emit `proofs/m11p3/limiter_diagnostics_24h.json`. Acceptance:
- Limited step count: ≤ 10% of 8,640 (vs M11.2's 8,640/8,640)
- Max cells limited per step: ≤ 10% of M11's 315k
- Mass residual: bounded, ≤ 0.05 (M11.2 was Infinity)

### AC6 — 24h pipeline runs to completion

Run `scripts/m7_daily_pipeline.py --hours 24 ...`. Emit pipeline + skill diff. Acceptance:
- 24 wrfouts produced
- All-finite-check PASS
- Pipeline NOT BLOCKED
- T2 RMSE measurable (regardless of value — fix proves stability first)

### AC7 — 100-step "parity" test passes (sanity)

`taskset -c 0-3 pytest -q tests/savepoint/test_dycore_100_steps.py` PASSES. Worker MUST note in the report that this test is tautological (agy finding) and document its limited value.

### AC8 — Worker report

`.agent/sprints/2026-05-28-m11p3-coordinated-dycore-fix/worker-report.md`. Verdict `M11P3_COMPLETE` if AC1-AC7 all pass; `M11P3_PARTIAL` with explicit gaps otherwise. **Include**: speedup numbers (we want to know how restored advection affects performance).

## Hard rules

1. **CPU pinning**: `taskset -c 0-3`.
2. **GPU usage**: YES — `--sandbox danger-full-access`.
3. **Files writable**: `src/gpuwrf/dynamics/**`, `src/gpuwrf/runtime/operational_mode.py` (just the `_rk_scan_step` advance_stage function — surgical), `proofs/m11p3/**`, `.agent/sprints/2026-05-28-m11p3-coordinated-dycore-fix/**`.
4. **Files NOT writable**: physics couplers (M12/M13 territory), BC code (M14 territory), comparator scripts (separate sprint), state contracts, governance.
5. **No remote push.**
6. **Manager repo ONLY**.
7. **Auto-notify on exit**: `tmux send-keys -t 0 "AGENT REPORT: m11p3 DONE exit=$?" Enter`.
8. **End with verdict**: `M11P3_COMPLETE` / `M11P3_PARTIAL` + headline (limiter drop %, 24h pipeline status, T2 RMSE if measurable).
