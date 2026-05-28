# Sprint Contract — M14: Lateral Boundary + Nesting Completeness

**Sprint ID**: `2026-05-28-m14-lateral-bc-completeness`
**Worker**: codex gpt-5.5 xhigh
**Branch**: `worker/gpt/m14-lateral-bc-completeness`
**Worktree**: `/tmp/wrf_gpu2_m14`
**Wall-time**: 6-18 h (target ≤ 1 day)
**GPU usage**: YES
**Sandbox**: `--sandbox danger-full-access`

## Why this sprint

The plan-critic round 1 (C5#2) and blinded planner (M5) both flagged lateral BC as incomplete. Current `apply_lateral_boundaries` applies U/V/theta/QVAPOR/PH/MU only; `BoundaryState` already sketches W/P/PB but they are not applied. M9 first-divergence at U/hr1 implies BC may contribute to wind divergence. M14 closes the boundary-completeness gap.

## Binding goal

A JAX-native GPU port of WRF v4 delivering Canary L2/L3 24-72 h RMSE on T2/U10/V10 **statistically equivalent** to CPU WRF v4 under **TOST** at predeclared margins on ≥30-case seasonal ensemble; ≥10× speedup preserved.

## Required inputs

1. `proofs/m9/divergence_map_v2.json` — wind divergence evidence
2. `src/gpuwrf/dynamics/boundary_apply.py` (or wherever `apply_lateral_boundaries` lives)
3. `src/gpuwrf/contracts/state.py` — `BoundaryState` definition
4. WRF reference: `dyn_em/module_bc.F` for boundary application order + relax-zone-width handling
5. Canary 20260521 `wrfbdy_d02` if available; otherwise documented as M14_BLOCKED

## Acceptance

### AC1 — Boundary variable completeness

`apply_lateral_boundaries` now covers all of: U, V, W, T (theta), QVAPOR, P (perturbation), PB (base), PH (perturbation geopotential), PHB (base geopotential), MU, MUB. State + BoundaryState extended with whichever leaves are missing.

### AC2 — Relax-zone width WRF-matched

Verify the relax-zone width matches WRF's `spec_bdy_width` setting (currently iter2 used spec_bdy_width=5). Apply the same width on every variable.

### AC3 — Boundary strip RMSE

For each variable above, compute the relative RMSE of GPU-applied boundary value vs decoded `wrfbdy_d02` boundary value. Emit `proofs/m14/boundary_strip_parity.json` schema:
```json
{
  "U": { "rel_rmse": <v>, "verdict": "PASS" if rel_rmse <= 1e-6 else "FAIL" },
  ...
}
```
At least 7 of 10+ variables PASS.

### AC4 — Interior-vs-boundary first-hour split

Re-run Canary 20260521 1h forecast. Compare GPU-vs-WRF hour-1 wrfout in two slices: (a) interior cells (>10 cells from any boundary), (b) boundary strip cells. Emit `proofs/m14/interior_vs_boundary_split.json`. Acceptance:
- Boundary strip RMSE on U drops vs M11 baseline (m11 v3 trace)
- Interior RMSE either holds or improves

### AC5 — 100-step parity preserved

`taskset -c 0-3 pytest -q tests/savepoint/test_dycore_100_steps.py` PASSES.

### AC6 — Harness re-verification

Re-run `scripts/run_diagnostic_harness.py --hours 1 --radiation-cadence-steps 999999`. `lateral_boundary` verdict remains `ACTIVE`. No regression on other operators.

### AC7 — Worker report

Standard format. Verdict `M14_COMPLETE` if AC1-AC6 all pass; `M14_PARTIAL` with explicit gaps otherwise.

## Hard rules

1. **CPU pinning**: `taskset -c 0-3`.
2. **GPU usage**: YES — `--sandbox danger-full-access`. Coordinate with parallel M11.2 + RRTMG-triage workers.
3. **Files writable**: `src/gpuwrf/dynamics/boundary_apply.py` or equivalent, `src/gpuwrf/contracts/state.py` (BoundaryState only — extend, don't rewrite), `proofs/m14/**`, `.agent/sprints/2026-05-28-m14-lateral-bc-completeness/**`.
4. **Files NOT writable**: dycore-internal (M11.2 territory), physics couplers, RRTMG, runtime/operational_mode aside from BC call site, governance.
5. **No remote push.**
6. **Manager repo ONLY**.
7. **Auto-notify on exit**: `tmux send-keys -t 0 "AGENT REPORT: m14 DONE exit=$?" Enter`.
8. **End with verdict**: `M14_COMPLETE` / `M14_PARTIAL` + headline boundary-strip pass count.
