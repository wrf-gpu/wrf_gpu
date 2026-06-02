# GPT RRTMG Topographic Shading Review

## Objective

Assess whether the GPU RRTMG path already represents WRF topographic slope/aspect/shadow SW radiation over steep d03 Canary terrain, and either close a real fidelity gap or document adequate fidelity.

## Finding

There is a real gap in the surface-energy radiation path. Gross GPU `SWDOWN` and `GLW` are mostly adequate against the d03 corpus state, but WRF's slope-adjusted `SWNORM` differs strongly from gross `SWDOWN` on steep daylight terrain.

Authoritative current-code oracle: `.agent/reviews/2026-06-02-gpt-rrtmg-toposhade-current-code-metrics.json`.

Steep daylight d03 cells: 1063 at 09Z/12Z/15Z/18Z. Predeclared tolerance was gross `SWDOWN` RMSE <= 20 W m-2 on daylight steep cells, `GLW` <= 20 W m-2 as context, and WRF topo signal > 20 W m-2 as a real surface-energy gap.

| valid time | gross GPU vs WRF `SWDOWN` RMSE | GPU gross vs WRF `SWNORM` RMSE | WRF `SWNORM-SWDOWN` signal RMSE | GPU vs WRF `GLW` RMSE |
|---|---:|---:|---:|---:|
| 2026-05-22 09Z | 9.64 | 107.47 | 105.85 | 1.96 |
| 2026-05-22 12Z | 25.87 | 71.23 | 57.67 | 8.28 |
| 2026-05-22 15Z | 24.08 | 69.28 | 59.09 | 2.79 |
| 2026-05-22 18Z | 10.38 | 116.14 | 114.89 | 5.33 |

Interpretation: matching gross `SWDOWN` is not enough for terrain fidelity. Pristine WRF keeps output `SWDOWN` gross and uses the surface-driver `TOPO_RAD_ADJ` result as `SWNORM` / scaled `GSW` for the surface-energy path.

## Fix

Changed `src/gpuwrf/physics/rrtmg_sw.py` only.

- Added `RRTMGSWTopographyState` and `RRTMGSWTopographicAdjustment`.
- Added `wrf_topographic_sw_correction_factor`, a direct port of pristine WRF `TOPO_RAD_ADJ`: latitude in degrees, declination/hour angle/slope/aspect in radians, `shadow_mask == 1`, WRF night/flat/all-diffuse/shadow branch behavior, no empirical clamp.
- Added `apply_wrf_topographic_sw_adjustment`, scaling surface down/up/absorbed SW by the WRF correction factor, matching WRF's `GSW = GSW * SWDOWN_teradj / SWDOWN` behavior.
- Exposed RRTMG direct/diffuse diagnostics: `surface_direct`, `surface_diffuse`, `surface_diffuse_fraction`.
- Added optional `topography=` to `solve_rrtmg_sw_column`. Default behavior remains gross WRF `SWDOWN` parity: `surface_down_topographic == surface_down` and correction factor is 1 unless terrain geometry is supplied.

## Commands Run

- `git log -1 --oneline --decorate`
- `nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader`
- `taskset -c 0-3 env OMP_NUM_THREADS=4 JAX_ENABLE_X64=true JAX_PLATFORM_NAME=cpu PYTHONPATH=src python - <<'PY' ...`
- `taskset -c 0-3 env OMP_NUM_THREADS=4 JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src pytest -q tests/test_m5_rrtmg_column_shapes.py tests/test_m5_rrtmg_transfer_solver.py tests/test_m5_rrtmg_tier1.py`
- `taskset -c 0-3 env OMP_NUM_THREADS=4 JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src python - <<'PY' ...`
- `git diff --check -- src/gpuwrf/physics/rrtmg_sw.py`

## Gate Result

Passed.

- Helper sanity: flat slope correction = 1; shadow correction = diffuse fraction; night correction = 1; sloped correction matches a manual evaluation of the WRF formula.
- Optional `topography=` solve path compiles and returns finite topographic fluxes with the expected shape.
- Focused pytest gate: `9 passed in 18.49s`.
- GPU availability check before JAX validation: 0% utilization, no model jobs; only desktop processes resident.

## Proof Objects Produced

- `.agent/reviews/2026-06-02-gpt-rrtmg-toposhade-current-code-metrics.json` - primary current-code d03 corpus-state assessment.
- `.agent/reviews/2026-06-02-gpt-rrtmg-toposhade-metrics.json` - supporting full-forecast d03 comparison; less authoritative because forecast/cloud drift is confounded.
- `.agent/reviews/2026-06-02-gpt-rrtmg-toposhade.md` - this handoff.

## Unresolved Risks

- Operational d02/d03 surface-energy parity still needs a coupling/surface integration step outside this lane's file ownership: pass latitude, declination, hour angle, slope, slope azimuth, and shadow mask into `RRTMGSWTopographyState`, then consume `surface_down_topographic` for the terrain-adjusted surface-energy path / `SWNORM`.
- Direct/diffuse partition is exposed from the existing kernel direct-transmittance path; I did not separately validate `surface_direct/surface_diffuse` against WRF `SWDDIR/SWDDIF` savepoints in this sprint.
- No full d02/d03 GPU forecast remeasure was run after coupling because coupling edits were explicitly out of scope.

## Next Decision Needed

Approve a follow-on integration lane, owned by coupling/surface files, to wire the new RRTMG topographic SW output into the surface-energy path and then rerun d02/d03 GPU `SWNORM`/`SWDOWN` parity when the GPU is free.
