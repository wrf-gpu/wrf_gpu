# Sprint Contract — M9.C: Theta-Convention + Comparator Audit (NEW, inserted before M11)

**Sprint ID**: `2026-05-28-m9c-theta-comparator-audit`
**Worker**: codex gpt-5.5 xhigh
**Branch**: `worker/gpt/m9c-theta-comparator-audit`
**Worktree**: `/tmp/wrf_gpu2_m9c`
**Wall-time**: 4-12 h (target 1 day)
**GPU usage**: YES — for re-running the trace after fix
**Sandbox**: `--sandbox danger-full-access`

## Why this sprint exists

M9 diagnostic (`proofs/m9/divergence_map.json`) found multi-source defects at hour 1 across most operational fields, but the theta mean RMSE of 75 K and theta max-max 345 K and SWDOWN max 1122 W/m² and HFX max 4105 W/m² are **too unphysical** to be pure model bugs. The dominant hypothesis is the **GPU JAX state stores theta in absolute Kelvin while WRF wrfout stores T as perturbation from a 300 K base reference**. If true: a one-line conversion in the comparator removes ~75 % of the apparent divergence and re-ranks the remaining defects. This must be resolved BEFORE M11 because M11's acceptance gate is "theta first-hour normalized RMSE vs WRF savepoint ≤ 1e-3" — and we currently don't know what the real number is.

## Binding goal

A JAX-native GPU port of WRF v4 delivering Canary L2/L3 24-72 h RMSE on T2/U10/V10 **statistically equivalent** to CPU WRF v4 under **TOST** at predeclared margins on ≥30-case seasonal ensemble; ≥10× speedup preserved.

## Objective

Three deliverables:

1. **Comparator audit**: read `scripts/operational_trace_compare.py` and the WRF wrfout variable conventions; identify every place where GPU-JAX and WRF wrfout could differ in (a) reference state / perturbation, (b) units, (c) vertical staggering / level indexing, (d) NaN / missing-data handling, (e) C-grid vs A-grid wind interpretation.

2. **Fixed comparator + re-run**: apply all confirmed corrections; re-run the trace on Canary 20260521 with the same wrfout reference.

3. **Updated divergence map**: write `proofs/m9/divergence_map_v2.json` with the post-correction numbers and an updated defect ranking. Update the manager-opus closure report to reflect the new diagnosis.

## Required inputs

1. `proofs/m9/operational_trace_hourly.json` — the v1 trace
2. `proofs/m9/divergence_map.json` — manager-opus v1 verdict + caveats list
3. `scripts/operational_trace_compare.py` — the comparator under audit
4. `scripts/m6b6_coupled_step_compare_1000.py` — also blocked, may share the issue
5. `src/gpuwrf/contracts/state.py` — what does the State store for theta? perturbation or absolute?
6. `src/gpuwrf/runtime/operational_mode.py` — what does the operational wrfout-equivalent writer store?
7. WRF source for reference: `dyn_em/solve_em.F` if present; otherwise consult WRF user-guide conventions for `T` vs `THETA`
8. `/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z/wrfout_d02_2026-05-21_19:00:00` — a single WRF wrfout file to inspect convention by `ncdump -h` or netCDF4

## Acceptance

### AC1 — `.agent/sprints/2026-05-28-m9c-theta-comparator-audit/comparator_audit.md`

Lists, for each of (theta, T, T2, U, V, U10, V10, W, PSFC, P, PB, PH, QVAPOR, SWDOWN, GLW, HFX, LH, PBLH, TSK, LU_INDEX):
- GPU side: where in `src/gpuwrf/` it lives, units, reference state, vertical index convention
- WRF side: corresponding wrfout variable name, units, reference state, vertical index convention
- Verdict: BITWISE_MATCH | CONVERTIBLE_WITH_FIX | METHOD_BUG | REAL_BUG
- Fix (if CONVERTIBLE_WITH_FIX or METHOD_BUG): exact one-liner

### AC2 — `scripts/operational_trace_compare.py` updated

The script in this branch applies every confirmed CONVERTIBLE_WITH_FIX correction. Code change in a single function (preferably one converter function per field). No unrelated refactoring.

### AC3 — `proofs/m9/divergence_map_v2.json`

Re-run the trace with the fixed comparator on Canary 20260521. Emit the new map with the same schema as v1. Compute:
- Per-field per-hour RMSE post-correction
- New first-divergence ranking
- A `corrections_applied` block listing each fix and its impact on the per-field RMSE
- A new `viability_verdict` (likely still VIABLE; the question is **how much** divergence remains)

### AC4 — `.agent/sprints/2026-05-28-m9c-theta-comparator-audit/worker-report.md`

Standard format. Verdict: `M9C_COMPLETE` if comparator is now trustworthy; `M9C_PARTIAL` if some convention questions are unresolved.

### AC5 — Existing tests regression-free

`taskset -c 0-3 pytest -q tests/savepoint/ tests/test_m6b6_coupled_step_parity.py` PASSES.

## Hard rules

1. **CPU pinning**: `taskset -c 0-3`.
2. **GPU usage**: ALLOWED for re-running the trace. ONE GPU instance.
3. **Files writable**: `scripts/operational_trace_compare.py`, `proofs/m9/**`, `.agent/sprints/2026-05-28-m9c-theta-comparator-audit/**`.
4. **Files NOT writable**: `src/**` (this is diagnostic + comparator only — no model code changes; if a real model bug is uncovered, the worker reports it for M11 but does not fix it here).
5. **No remote push.**
6. **Manager repo ONLY**.
7. **Auto-notify on exit**: `tmux send-keys -t 0 "AGENT REPORT: m9c DONE exit=$?" Enter`.
8. **End with verdict**: `M9C_COMPLETE` / `M9C_PARTIAL` + headline reduction in apparent divergence.

## What success looks like

Best case: theta convention fix drops theta mean RMSE from 75 K to <5 K, T2 mean RMSE drops from 43 K to ~3-5 K (close to iter2 numbers), HFX peak drops from 4105 to ~500-1000 W/m². Then the operational pipeline is producing skill ~iter2 quality and the remaining gap is the real M11-M14 work. Worst case: the corrections shrink the apparent divergence somewhat but real defects remain — M11-M14 still proceed with cleaner targets.
