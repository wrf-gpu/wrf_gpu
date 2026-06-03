# v0.6.0 PBL GPU-op handoff — YSU + ACM2 jax.lax.scan-traceable rewrite + scan-wire

Date: 2026-06-03
Author: Opus 4.8 (1M) worker
Branch: `worker/opus/v060-pbl-gpuop` (base `382070f` = the scan-wire commit)
Environment: JAX CPU only, cores 0-3 (`taskset -c 0-3`). NO GPU — parity +
scan-smoke are CPU/trace-time; the GPU multi-config forecast gate is MANAGER-held.

## Objective

The v0.6.0 YSU (bl_pbl=1) + ACM2 (bl_pbl=7) PBL ports passed per-scheme WRF
savepoint parity but their kernels were single-column HOST-NumPy (`_scalar` +
Python level loops) — NOT `jax.lax.scan`-traceable, so they were fail-closed in the
operational GPU scan. This sprint makes them GPU-operational: a traceable / vmap
rewrite, re-verified parity (no regression), and scan-wired into the forecast loop.

## What was done

### 1. jax.lax.scan-traceable / vmap-batched rewrite (no clamp/mask)
Both kernels are now a **1:1 transcription of the host-NumPy reference into pure
`jnp` / `jax.lax`**, batched over `(ncol, nz)`:
- **YSU** (`src/gpuwrf/physics/pbl_ysu.py`): the bulk-Richardson PBL-height searches
  (`_first_pbl_guess`, the brcr re-scan) → `jax.lax.scan` with a freeze-on-stable
  carry; the non-local K-profile + counter-gradient (hgamt/hgamq/hgamu/hgamv) and
  the entrainment block → masked `jnp.where` per level; the 4 implicit
  vertical-diffusion tridiagonals (heat/qv/u/v) → a **scan-based Thomas solver**
  (`_thomas_scan`) matching `tridin_ysu`/`tridi2n` indexing exactly. New entries:
  `_ysu_column_traceable` (single column), `ysu_columns` (vmapped over the grid).
- **ACM2** (`src/gpuwrf/physics/pbl_acm2.py`): the PBL-height diagnosis (ksrc/kmix/
  kpblh break-searches) → masked argmax; `_eddyx` → `jnp.where` branches; the WRF
  `TRI` and bordered-band `MATRIX` solvers → static (nz-bounded) Python loops that
  unroll at trace time (`_tri_solve_1based_traceable`, `_matrix_solve_1based_
  traceable`); the **data-dependent semi-implicit substep count `nlp`** → a
  fixed-max (`_NLP_MAX=16`) `jax.lax.scan` that freezes each column after its own
  `nlp` (`jnp.where(i<nlp, advanced, vci)` — exact for any `nlp<=_NLP_MAX`). New
  entries: `_acm2_column_traceable`, `acm2_columns`.

The host-NumPy reference (`_ysu_numpy` / `_acm2_numpy`) is retained ONLY for the
cross-check. `step_{ysu,acm2}_column` (the parity-test entry) now route through the
traceable kernel, so the per-scheme parity test exercises the GPU-operational path.

### 2. Parity re-verified — NO regression
- `tests/test_v060_pbl_ysu.py` + `tests/test_v060_pbl_acm2.py` → **4 passed**, both
  reports **verdict=PASS** on all 6 cases each, SAME predeclared tolerances.
- Trace-vs-host-NumPy cross-check (all 6 cases each): **max abs diff ~1.9e-15**
  (machine precision) → traceable == prior-host == WRF. The committed report JSONs
  differ from the host version only in sub-1e-19 XLA-vs-NumPy reordering noise in
  the `max_abs`/`abs_error_by_level` fields (errors still ~1e-12, tol 2e-6).
- TWO real transcription bugs found + fixed during the ACM2 rewrite (both were my
  own, caught by the host cross-check, NOT masked): (a) `bi[k]=1` for the above-PBL
  band must be UNCONDITIONAL (it is outside WRF's `if noconv` block) — I had gated
  it behind noconv; (b) `ei[1]`/`ai[2]` surface coupling is UNCONDITIONAL too. After
  both fixes the local-TRI cases (3,4,5) went from ~1e16 error to ~1e-15.

### 3. Scan-wired into the operational forecast loop (gpu_runnable now genuine)
- NEW `coupling/scan_adapters.py` adapters `ysu_pbl_adapter` / `acm2_pbl_adapter`
  + `PBL_SCAN_ADAPTERS = {1: ysu, 7: acm2}`. The adapter re-derives the per-cell
  surface forcing the kernels need (YSU: HFX/QFX/BR/psim/psih/U10/V10/ZNT; ACM2:
  HFX/QFX/pblh/wspd) via the SAME revised-MM5 surface layer the scan already runs
  (`surface_layer_with_diagnostics`) — the exact forcing the savepoint-parity kernel
  was validated on; no host transfer, fully traceable. Momentum is coupled as the
  A-grid PBL increment A2C-averaged onto the original C-grid faces (WRF
  `add_a2c_u`/`add_a2c_v`), mirroring `_state_from_mynn_output`; theta/qv direct.
- `runtime/operational_mode.py`: PBL slot now dispatches `bl_pbl` via
  `PBL_SCAN_ADAPTERS` (static branch, no per-step lax.cond); `_SCAN_WIRED_OPTIONS`
  bl_pbl = `(0,1,5,7)`; YSU/ACM2 removed from `_SCAN_UNWIRED_REASON`; fail-closed
  message updated. `bl=5` MYNN unchanged (default path byte-for-byte).
- `coupling/physics_dispatch.py`: PBL `gpu_runnable=True` is now GENUINE (comment
  corrected) — the host-NumPy single-column path was replaced and
  `_resolve_operational_suite` no longer fails closed on bl=1/7.
- Scan smoke `proofs/v060/pbl_gpuop_smoke.py` → **all_pass=True**: both adapters
  **JIT-compile** (the GPU-op proof — host-NumPy couldn't trace at all), execute,
  stay finite, conserve (column-water rel change 0.07–0.1%), and
  `_resolve_operational_suite` now ACCEPTS bl=1/7.

### 4. Stale-test fix
`tests/test_noahmp_checkpoint_v2.py::test_format_version_is_2` now asserts
`FORMAT_VERSION == 3` (the v0.6.0 integration bumped checkpoint 2→3). Passes.

### "Do NOT do" items — confirmed cleanly fail-closed (unchanged)
Noah-classic (sf_surface=2), GF (cu=3), Tiedtke (cu=6/16) still reject loudly in
`_resolve_operational_suite` / the scanwire smoke fail-closed check (verified).

## Files changed
- `src/gpuwrf/physics/pbl_ysu.py` — `_thomas_scan`, `_first_pbl_guess_traceable`,
  `_interp_pblh_traceable`, `_ysu_column_traceable`, `ysu_columns`; `step_ysu_column`
  routes through the traceable kernel.
- `src/gpuwrf/physics/pbl_acm2.py` — `_mean_first_k`, `_diagnose_pbl_height_traceable`,
  `_eddyx_traceable`, `_tri_solve_1based_traceable`, `_matrix_solve_1based_traceable`,
  `_mix_acm_values_traceable`, `_acm2_column_traceable`, `acm2_columns`;
  `step_acm2_column` routes through the traceable kernel. `_NLP_MAX=16`.
- `src/gpuwrf/coupling/scan_adapters.py` — `ysu_pbl_adapter`, `acm2_pbl_adapter`,
  `_pbl_surface_forcing`, `_apply_pbl_increment`, `PBL_SCAN_ADAPTERS`; module docstring.
- `src/gpuwrf/runtime/operational_mode.py` — PBL-slot dispatch; `_SCAN_WIRED_OPTIONS`;
  `_SCAN_UNWIRED_REASON`; fail-closed message; PBL_SCAN_ADAPTERS import.
- `src/gpuwrf/coupling/physics_dispatch.py` — PBL entries comment (gpu_runnable genuine).
- `tests/test_noahmp_checkpoint_v2.py` — FORMAT_VERSION 2→3 assertion.
- `proofs/v060/scanwire_smoke.py` + `gen_scanwire_report.py` — YSU/ACM2 now wired
  (moved from fail-closed to accepted; 8 wired / 3 fail-closed).
- NEW `proofs/v060/pbl_gpuop_smoke.py`, `gen_pbl_gpuop_report.py`,
  `pbl_gpuop_smoke.json`, `pbl_gpuop_report.json`.
- Regenerated `ysu_/acm2_savepoint_parity_report.json`, `scanwire_report.json`,
  `scanwire_smoke.json`.

## Commands run
- `pytest tests/test_v060_pbl_ysu.py tests/test_v060_pbl_acm2.py` → 4 passed (PASS/PASS).
- trace-vs-host cross-check (6+6 cases) → max abs ~1.9e-15.
- `python proofs/v060/pbl_gpuop_smoke.py` → all_pass=True (JIT, finite, conserve).
- `pytest` v060 PBL/dispatch/sfclay/KF + checkpoint + restart-full-carry → 31 passed.
- `python proofs/v060/gen_pbl_gpuop_report.py` → all_pass=True.

## Proof objects
- `proofs/v060/pbl_gpuop_report.json` — consolidated: parity STILL passes
  post-rewrite, scan-smoke passes, YSU/ACM2 now scan-wired + gpu_runnable. all_pass=True.
- `proofs/v060/pbl_gpuop_smoke.json` — per-scheme JIT smoke detail.
- `proofs/v060/{ysu,acm2}_savepoint_parity_report.json` — verdict=PASS (traceable path).

## Unresolved risks / next decision
1. **GPU multi-config forecast gate (MANAGER):** numerical equivalence of the
   YSU/ACM2 PBL combos vs CPU-WRF (per-lead bias/RMSE) is the manager's single GPU
   job; this CPU sprint validates traceability + per-column parity + finite/conserve.
2. **ACM2 `_NLP_MAX=16` substep cap:** exact for any column with `nlp<=16`. The
   savepoint cases use `nlp=2`; a real grid's near-surface CFL could in principle
   exceed 16 (the kernel silently runs only 16 of a larger `nlp` — UNDER-mixing, not
   a blow-up). Recommend the GPU gate log the max `nlp` over the domain; raise the
   cap if it approaches 16. The unrolled MATRIX makes the ACM2 trace compile slow
   (~2 min in the parity test); cached on the GPU forecast path.
3. **Surface-forcing assembler:** the PBL adapter re-derives forcing via the
   revised-MM5 surface layer + uses lowest-level pressure as the PSFC proxy and a
   1000 m `pblh_initial` seed for ACM2 (it diagnoses its own height immediately).
   Reasonable for the scan-wire; the GPU gate vs CPU-WRF validates the coupling and
   may want the true PSFC / a persisted PBLH leaf.
4. The full `run_forecast_operational` end-to-end requires a GPU (the dycore
   `Tendencies.zeros` is GPU-gated), so end-to-end was validated at the adapter
   level (the exact scan-body PBL function, JIT-compiled) + dispatch/resolve, not a
   full CPU forecast.
