# Sprint Contract — M11.1: Dycore p_perturbation + ph_perturbation Activity Triage

**Sprint ID**: `2026-05-28-m11p1-dycore-pressure-geopotential-triage`
**Worker**: codex gpt-5.5 xhigh
**Branch**: `worker/gpt/m11p1-dycore-p-ph-triage`
**Worktree**: `/tmp/wrf_gpu2_m11p1`
**Wall-time**: 4-12 h (target ≤ 1 day)
**GPU usage**: YES (1h harness rerun + potentially 24h pipeline)
**Sandbox**: `--sandbox danger-full-access`
**Per plan-critic round 2 PC2.10**: single highest-impact next dispatch after M17

## Why this sprint exists

The diagnostic harness smoke (`proofs/diagnostic_harness/diagnostic_report_smoke_3step.json`) found `dycore_rk3` verdict `NOISY_ZERO` on `p_perturbation` and `ph_perturbation` — these prognostic fields have ZERO delta across the 3-step run. The plan-critic round 2 ruled this Phase-B-blocking: if they're expected to update, the dycore has a silent identity path; if they're diagnostic/derived, the harness expectation is wrong. Either resolution is required before Phase B can close.

## Binding goal

A JAX-native GPU port of WRF v4 delivering Canary L2/L3 24-72 h RMSE on T2/U10/V10 **statistically equivalent** to CPU WRF v4 under **TOST** at predeclared margins on ≥30-case seasonal ensemble; ≥10× speedup preserved.

## Required inputs

1. `proofs/diagnostic_harness/diagnostic_report_smoke_3step.json` — the smoking gun
2. `src/gpuwrf/diagnostics/comprehensive_harness.py` — what the harness expects (USE, don't modify)
3. `src/gpuwrf/dynamics/**` — dycore source
4. `src/gpuwrf/contracts/state.py` — state contract for p_perturbation, ph_perturbation, p_base, ph_base leaves
5. `src/gpuwrf/runtime/operational_mode.py` — operator sequence in `_physics_boundary_step`
6. WRF source reference: how WRF treats p_pert / ph_pert in `dyn_em/solve_em.F` + `dyn_em/module_em.F`

## Acceptance

### AC1 — Triage decision

`.agent/sprints/2026-05-28-m11p1-dycore-pressure-geopotential-triage/triage.md`:
- Read the WRF reference for p_perturbation and ph_perturbation semantics
- Determine whether they should be prognostic (RK3 updates them every step) or diagnostic (derived post-step from other state)
- Match against the GPU JAX state contract: what does `state.p_perturbation` and `state.ph_perturbation` represent?
- Decision: `PROGNOSTIC_BUG_FIX_DYCORE` | `DIAGNOSTIC_FIX_HARNESS_EXPECTATION` | `MIXED_NEED_BOTH_FIXES`

### AC2 — Apply the chosen fix

**If `PROGNOSTIC_BUG_FIX_DYCORE`**: Fix the dycore update path for these fields. Minimal diff in `src/gpuwrf/dynamics/**`. Document the line that was previously skipping the update.

**If `DIAGNOSTIC_FIX_HARNESS_EXPECTATION`**: Modify `src/gpuwrf/diagnostics/comprehensive_harness.py` (this IS allowed for M11.1 alone, override of normal "don't modify harness" rule) to mark `p_perturbation` and `ph_perturbation` as DIAGNOSTIC_OK (a new verdict) rather than expecting them to be ACTIVE under dycore_rk3.

**If `MIXED_NEED_BOTH_FIXES`**: Do both, document why.

### AC3 — Harness re-verification

Re-run `scripts/run_diagnostic_harness.py` with 3-step horizon. Emit `proofs/m11p1/diagnostic_report_after_fix.json`. Acceptance:
- `dycore_rk3` verdict for p_perturbation and ph_perturbation is no longer `NOISY_ZERO` — must be `ACTIVE` (if prognostic fix), `PASSIVE_OK` / `DIAGNOSTIC_OK` (if harness expectation fixed), or unambiguous documented status.
- No other operator regresses.

### AC4 — 100-step parity preserved

`taskset -c 0-3 pytest -q tests/savepoint/test_dycore_100_steps.py` PASSES.

### AC5 — Documentation

Update `tests/savepoint/README.md` if M11.1 added a new verdict to the harness. Update `src/gpuwrf/contracts/state.py` docstring on p_perturbation / ph_perturbation if their semantics needed clarification.

### AC6 — Worker report

`.agent/sprints/2026-05-28-m11p1-dycore-pressure-geopotential-triage/worker-report.md`. Verdict: `M11P1_COMPLETE` if AC1-AC5 all pass; `M11P1_PARTIAL` otherwise.

## Hard rules

1. **CPU pinning**: `taskset -c 0-3`.
2. **GPU usage**: YES — `--sandbox danger-full-access`. Coordinate with parallel m11 + m17 workers (3 codex total, at cap).
3. **Files writable**: `src/gpuwrf/dynamics/**` (if prognostic fix), `src/gpuwrf/diagnostics/comprehensive_harness.py` (if harness expectation fix — special override for M11.1 ONLY), `src/gpuwrf/contracts/state.py` docstring only, `proofs/m11p1/**`, `tests/savepoint/README.md`, `.agent/sprints/2026-05-28-m11p1-dycore-pressure-geopotential-triage/**`.
4. **Files NOT writable**: physics couplers (M11/M12/M13/M17 territory), runtime/operational_mode (M11 territory), BC, governance.
5. **No remote push.**
6. **Manager repo ONLY**.
7. **Auto-notify on exit**: `tmux send-keys -t 0 "AGENT REPORT: m11p1 DONE exit=$?" Enter`.
8. **End with verdict**: `M11P1_COMPLETE` / `M11P1_PARTIAL` + headline (decision class + harness verdict post-fix).
