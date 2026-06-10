# V0.14 Fable PSFC Moist Pressure-State Closure — Worker Report

Date: 2026-06-10. Worker: Fable high. Sprint:
`.agent/sprints/2026-06-10-v014-fable-psfc-moist-pressure-closure/sprint-contract.md`.
CPU-only; the active 72h GPU run was never touched.

## Decision: FIXED (PSFC, WRF-exact diagnostic) + FORMALLY_BOUNDED (3D P state)

## Objective

Close the fixed-Canary `PSFC` vapor-light residual (h1 `-154.9 Pa` worsening
to `-229.9 Pa` by h10 per the manager addendum) with a WRF-faithful fix, or
prove it unfixable in v0.14 scope.

## Root cause (proven, two layers)

1. **PSFC formula bug (FIXED).** WRF's runtime `PSFC` is NOT an extrapolation
   of the nonhydrostatic `P+PB`. It is `p8w(kts)` where the surface driver is
   called with `P8W = grid%p_hyd_w` — the **moist hydrostatic** integral
   `p_top + sum_k (1+qtot_k)*(c1h_k*(MU+MUB)+c2h_k)*(-dnw_k)`, qtot over ALL
   moist species. Anchors: `module_surface_driver.F:1988`,
   `module_first_rk_step_part1.F:1400`,
   `module_big_step_utilities_em.F:4946-4958` (pristine WRF tree
   `/home/enric/src/wrf_pristine/WRF`; the contract's Gen2 WRF path does not
   exist on this box). Formula proven on CPU truth to **<= 0.18 Pa RMSE** at
   h1/h4/h10/h24 — this also resolves GPT's "~14 Pa formula gap" risk (that
   gap belonged to the extrapolation diagnostic, not WRF's PSFC).
   The GPU extrapolated `P+PB` instead — and because of layer 2 the GPU
   pressure state carries no vapor load, so PSFC missed the full vapor column.
2. **Dry-balanced 3D pressure state (BOUNDED, next sprint).** GPU `P+PB(k0)`
   sits on its own DRY hydrostatic column (-8.2 Pa h1) while CPU sits on its
   MOIST column (-13.5 Pa). Code anchor: the operational acoustic w-equation
   uses `dry_cqw`/`pg_buoy_w_dry` (`operational_mode.py:118,1209,1274,1483,1657`;
   `dynamics/core/advance_w.py`) — `cq1=1, cq2=0` — omitting WRF `calc_cq`
   (`:787-899`) + moist `pg_buoy_w` (`:2419-2500`) vapor load. The solver
   infrastructure already accepts a `cqw` field (`core/acoustic.py:707`), so
   the moist coupling is a wiring + acoustic-oracle + GPU-gate sprint; it
   changes prognostic `P/PH/W` and was deliberately NOT bundled here. It
   explains the remaining 3D `P` lane (h10 `P` RMSE 65.7 Pa).

## Fix + measured effect (offline ablation on the fixed run, d02)

| Lead | PSFC bias/RMSE current | PSFC bias/RMSE post-fix |
|---:|---|---|
| h1  | -154.9 / 157.0 | **+51.6 / 57.8** |
| h4  | -185.3 / 186.7 | **+18.3 / 35.5** |
| h10 | -229.9 / 231.3 | **-10.8 / 28.6** |
| h24 | -294.9 / 296.1 | **-58.1 / 64.2** |

Gate `h1 <= 120 Pa`: MET. Post-fix PSFC now tracks the (already-improving)
dry-mass `MU` lane + a slowly growing GPU vapor-column dry bias
(-1.3 -> -48.1 Pa h1->h24); both stay visible to the comparator — no masking.
`MU/P/PH/U/V/T/QVAPOR` untouched by construction (diagnostic-only change).

## Files changed

- `src/gpuwrf/runtime/operational_mode.py` — `_psfc_from_state(state, metrics)`
  = WRF `p_hyd_w(kts)` integral; caller passes `namelist.metrics`. Constant
  memory/column, inside the existing jitted M9 snapshot, no timestep-loop
  transfers.
- `src/gpuwrf/io/wrfout_writer.py` — `PSFC` fallback uses the same integral
  when `grid.metrics` is resident; metric-less synthetic states keep the old
  extrapolation.
- `scripts/diag/d03_pressure_knockout.py` — updated to the new signature
  (direct consumer of the changed function).
- NEW `tests/test_v014_psfc_moist_hydrostatic.py` (2 tests: function vs the
  verbatim WRF recurrence; writer fallback end-to-end through
  `write_wrfout_netcdf`).
- NEW `proofs/v014/psfc_moist_pressure_state_closure.{py,md,json}`.

## Commands run

See proof §5. Summary: proof script (budget+ablation, h1/h4/h10/h24); new
tests 2/2 PASS; focused suite `test_m7_netcdf_writer` `test_m7_daily_pipeline`
`test_async_wrfout_equiv` `test_auxhist_multistream` `test_auxhist_stream`
24 passed/1 skipped; `python -m json.tool` OK; full
`py_compile src tests proofs` OK; `git diff --check` OK.
Pre-existing (NOT mine, fails identically on unmodified tree):
`test_rrtm_lw_operational_wiring.py::test_dispatch_routes_ra_lw1_to_rrtm_and_differs_from_rrtmg`
(XLA:CPU AOT-cache machine-feature NaN class).

## Proof objects

- `proofs/v014/psfc_moist_pressure_state_closure.md` / `.json` / `.py`
- this report

## Unresolved risks

1. The live 72h run still emits old-formula PSFC; outputs produced before a
   relaunch/regeneration keep the vapor-light floor. Post-fix expectation is
   the ablation column above; a short GPU h1/h4 check after merge confirms.
2. 3D `P/PH` remains dry-balanced (layer 2) until the moist-cqw dynamics
   sprint; `P` k0 bias ~-212 Pa at h10 persists and is correctly NOT hidden.
3. `MU` lane is slightly negative at h24 (bias -9.8 Pa) and the GPU vapor
   column dries ~-48 Pa by h24 vs CPU — separate state lanes the comparator
   tracks; they now bound the post-fix PSFC error.

## Next decision needed

1. Manager review + merge of this diff; schedule the short GPU h1/h4
   validation (and decide whether the active 72h characterization run should
   be relaunched with the fix).
2. Open the moist-cqw w-equation dynamics sprint (calc_cq + moist pg_buoy_w +
   cqw threading; acoustic savepoint oracle + GPU gate) to close the 3D
   pressure-state lane before grid-parity promotion / Switzerland.
