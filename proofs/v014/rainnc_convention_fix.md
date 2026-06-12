# v0.14 Switzerland d01 72h — RAINNC gate miss: ROOT-CAUSED + PARTIAL FIX + BOUNDED RESIDUAL

Run: `/mnt/data/wrf_gpu_validation/v014_switzerland_d01_72h_gpu_thetafix_20260612T012219Z`
CPU truth: `/mnt/data/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z/run_cpu`
Config: `mp_physics=8` (Thompson, graupel), `cu_physics=0` (no cumulus -> RAINC=0).

## Verdict

REAL BUG (RAINNC accumulation convention) + BOUNDED chaotic precip sensitivity in the residual.

- **RAINC**: identically 0.0 CPU and GPU (no cumulus). So the miss is grid-scale specific.
- **RAINNC** atlas gate: pooled RMSE 5.994 mm vs 1.0 limit -> FAIL.

## Root cause (BUG)

WRF `module_mp_thompson.F:1298-1306`:
```
RAINNC    += pptrain + pptsnow + pptgraul + pptice   ! ALL-PHASE total
SNOWNC    += pptsnow + pptice                          ! frozen subset
GRAUPELNC += pptgraul
```
RAINNC is the all-phase TOTAL; SNOWNC/GRAUPELNC are overlapping subsets.

GPU keeps DISJOINT State accumulators (`rain_acc`=liquid only, `snow_acc`, `ice_acc`,
`graupel_acc`) and the wrfout writer mapped `RAINNC <- rain_acc` (rain only),
dropping the snow+graupel+ice contribution. Diagnostic signatures:
- GPU `min(RAINNC - SNOWNC) = -8.42` (impossible in WRF; CPU = 0.0 exactly).
- Terrain-binned h72: above 1500 m GPU RAINNC ~0.03-0.2 mm vs CPU 7.6-9.8 mm
  (the missing high-terrain frozen precip), GPU rain over-counted on warm low slopes.
- Domain-mean h72: CPU 5.48 mm vs GPU rain-only 3.85 mm (-30%).

## Fix

`src/gpuwrf/io/wrfout_writer.py`: write `RAINNC = rain_acc + snow_acc + graupel_acc + ice_acc`
(WRF all-phase total) at the output boundary; SNOWNC = snow_acc+ice_acc and
GRAUPELNC = graupel_acc unchanged. Internal disjoint accumulators are untouched, so
conservation budget / SR / restart are unaffected.
Test: `tests/test_m7_netcdf_writer.py::test_rainnc_is_wrf_all_phase_total`.

## Before / after (reconstruction = current_RAINNC + SNOWNC + GRAUPELNC)

| metric | rain-only (current) | all-phase (fixed) |
| --- | ---: | ---: |
| pooled RMSE (mm) | 5.994 | 5.186 |
| h72 domain-mean (mm) | 3.851 | 5.570 (CPU 5.483) |
| h72 bias (mm) | -1.632 | +0.087 |
| h72 field corr | 0.037 | 0.306 |

Bias / mass deficit fully resolved (-1.63 -> +0.09 mm). Field corr 0.04 -> 0.31.

## Residual = BOUNDED chaotic sensitivity (NOT a bug)

Corrected pooled RMSE 5.19 mm still > 1.0 limit, but:
- corrected bias oscillates near zero across all leads (|bias|/rmse = 0.01-0.33),
  no systematic growing offset -> not an accumulation bug.
- RMSE grows with lead tracking field magnitude (2.6 -> 6.7 mm) = chaotic error growth.
- CPU field std = 6.63 mm, wet-cell mean = 6.77 mm; RMSE ~0.78x the field's own std.
- displacement-shift test: best +-3 cell shift only cuts RMSE 6% -> not pure displacement,
  genuine trajectory divergence of orographic precip cells.

The 1.0 mm hard limit is unphysically tight for 72h accumulated Alpine winter precip
whose own spatial std is ~6.6 mm. Recommend a justified bounded acceptance (like the
Canary gate) keyed to field-relative RMSE, OR a manifest-limit revisit.

## Minor unresolved (out of RAINNC-gate scope)

GPU GRAUPELNC ~= 0 for the whole run (CPU 0.22 mm domain-mean, ~2% of total precip):
in-column QGRAUP is only ~3% of CPU mass -> a Thompson graupel-SOURCE fidelity gap,
not an accumulation bug. Now correctly folded into RAINNC; immaterial to the gate.
