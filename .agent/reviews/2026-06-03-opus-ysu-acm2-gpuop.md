# YSU(1) + ACM2(7) PBL GPU-operationalization — verification + proof

**Date:** 2026-06-03
**Branch:** `worker/opus/v060-ysu-acm2-gpuop` (base `e998250` trunk-0.9.0)
**Author:** Opus implementer lane
**Scope:** bl_pbl_physics = 1 (YSU) and 7 (ACM2) ONLY. MYJ(2)/Janjic(2) untouched.

## Objective

Make the YSU/ACM2 PBL kernels GPU-operational (jit/vmap-traceable, scan-wired)
while preserving the proven WRF-faithful savepoint parity, then produce the proof
objects.

## Key finding: the rewrite already existed in the base — the gap was *verification*

The base commit `9af2813 [v060] YSU+ACM2 jax.lax.scan-traceable rewrite + scan-wire
(GPU-operational)` had already:

- Rewritten the host-NumPy single-column kernels as pure-JAX traceable kernels:
  `pbl_ysu._ysu_column_traceable` / `pbl_acm2._acm2_column_traceable`, with
  vmap-batched entries `ysu_columns` / `acm2_columns` and scan-based Thomas /
  bordered-band tridiagonal solves. The host-NumPy references (`_ysu_numpy`,
  `_acm2_numpy`) are retained for cross-check only.
- Repointed the single-column public entries `step_ysu_column` / `step_acm2_column`
  to the traceable kernels (so the savepoint-parity tests already exercise the GPU-op
  path).
- Wired both into the operational scan: `coupling.scan_adapters.PBL_SCAN_ADAPTERS =
  {1: ysu_pbl_adapter, 7: acm2_pbl_adapter}`, dispatched by
  `runtime.operational_mode` PBL slot (line ~2133) on the static `bl_pbl_physics`.
- Updated `_resolve_operational_suite` GPU-runnable matrix to accept bl_pbl in
  {0,1,5,7} and `io/namelist_check.py` to list YSU/ACM2 as implemented.

So the task became: **independently re-verify the parity has not regressed, prove the
traceable path equals the WRF-validated reference, fill the missing transcription
proof + per-scheme GPU-op deliverables, and finish the registry status.**

## What I did

1. Re-ran the WRF-oracle savepoint parity (the tests regenerate the reports from the
   traceable `step_*_column` path) for all 6 stable/unstable/neutral cases each.
   Both **PASS** with the unchanged predeclared fp64 tolerances. Git diff on the
   regenerated reports is empty — committed reports already reflect the traceable code.
2. **Added the missing transcription-fidelity proof** `proofs/v060/pbl_trace_vs_host.py`
   (+ `.json`): runs BOTH the host-NumPy reference AND the traceable kernel on the
   identical savepoint inputs and reports the worst abs/rel diff per field. This backs
   the load-bearing claim that traceable == host-NumPy == WRF (the gpuop report
   previously hardcoded `1.9e-15` "verified separately").
3. Re-ran the GPU-op smoke `proofs/v060/pbl_gpuop_smoke.py` — both adapters
   jit-compile, execute in the scan, stay finite, conserve column water.
4. **Added per-scheme GPU-op deliverables** `proofs/v060/{ysu,acm2}_gpuop_savepoint_parity.json`
   (+ generator `gen_pbl_gpuop_savepoint_parity.py`): explicit GPU-op verdict wrapping
   the WRF-oracle parity + the trace cross-check, asserting `tolerances_loosened=False`.
5. Wired the real trace artifact into `gen_pbl_gpuop_report.py` (no more hardcoded
   number) and added `trace_ok` to its `all_pass`.
6. **Registry:** bumped YSU/ACM2 `SchemeOption.status` from `accepted` to `implemented`
   to match the MYNN GPU-wired template. `status` is a freeze-artifact label, not
   consumed by dispatch (which uses `PBL_SCAN_ADAPTERS` + the GPU-runnable matrix); no
   test asserts on it.

## Results (all CPU JAX fp64, cores 0-3, no GPU)

| Proof | Verdict |
|---|---|
| YSU savepoint parity vs unmodified WRF `module_bl_ysu.F` (6 cases, traceable path) | **PASS** |
| ACM2 savepoint parity vs unmodified WRF `module_bl_acm.F` (6 cases, traceable path) | **PASS** |
| Trace-vs-host transcription (YSU worst_abs 1.49e-17, half cases bitwise 0) | **PASS** |
| Trace-vs-host transcription (ACM2 worst_abs 1.89e-15; kpbl + noconv exact) | **PASS** |
| GPU-op smoke (jit adapter executes / finite / conserves; suite accepts bl=1,7) | **PASS** |
| Consolidated `pbl_gpuop_report.json` all_pass | **PASS** |
| Test suite (ysu+acm2 parity, dispatch, namelist_check, interfaces): 25 tests | **PASS** |

Worst savepoint residuals (both within predeclared tol via the abs-OR-rel rule):
- YSU: EXCH_H case 2 max_abs 7.6e-3 (max_rel 1.2e-5 << 2e-3 rel; EXCH_H scale ~hundreds).
- YSU: RTHBLTEN case 5 max_rel 1.2e5 is a near-zero-field artifact (scale 1e-12);
  binds on abs (max_abs 1.19e-7 < 2e-6). Genuine pass, not loosened.
- ACM2: EXCH_H case 2 max_abs 3.4e-3 (max_rel 8.2e-6 << 2e-3 rel).

No clamps, no masks-to-pass, no JAX-vs-JAX self-compare, no synthetic happy-path, no
loosened tolerance. The arbiter is the unmodified pristine-WRF single-column module
oracle; the trace cross-check is a transcription check, explicitly labeled as such.

## GPU-scan wired

- YSU bl_pbl=1: YES (`ysu_pbl_adapter` in `PBL_SCAN_ADAPTERS`, operational dispatch).
- ACM2 bl_pbl=7: YES (`acm2_pbl_adapter` in `PBL_SCAN_ADAPTERS`, operational dispatch).

## Unresolved risk

- **No coupled GPU multi-config forecast gate vs CPU-WRF** for the YSU/ACM2 PBL combos
  (per-lead bias/RMSE). This sprint is CPU-only / isolated savepoint + transcription +
  jit smoke; the end-to-end coupled GPU gate is MANAGER-scheduled (one GPU job).
  Risk: the adapter surface-forcing re-derivation (revised-MM5 sfclay) and A2C
  momentum coupling are validated only structurally (smoke), not against a coupled
  CPU-WRF run. The per-column kernel itself is WRF-parity-proven.
- The `_acm2_column_traceable` data-dependent semi-implicit substep count runs a
  fixed-max `jax.lax.scan` with per-column masking; correct on the 6 oracle cases but
  the max-substep ceiling should be revisited if a coupled run hits deep-convective
  ACM2 columns with more substeps than the cases exercised.

## Files changed

- `src/gpuwrf/contracts/physics_registry.py` — YSU/ACM2 status -> implemented.
- `proofs/v060/pbl_trace_vs_host.py` (NEW) + `pbl_trace_vs_host.json` (NEW).
- `proofs/v060/gen_pbl_gpuop_savepoint_parity.py` (NEW) +
  `{ysu,acm2}_gpuop_savepoint_parity.json` (NEW).
- `proofs/v060/gen_pbl_gpuop_report.py` — reads real trace artifact.
- `proofs/v060/pbl_gpuop_report.json` — regenerated.
- `.agent/reviews/2026-06-03-opus-ysu-acm2-gpuop.md` (this file).

Kernel files `pbl_ysu.py` / `pbl_acm2.py` and scan/dispatch wiring were already correct
in the base; not modified.
