# Noah-MP Snow (Sprint S3) — Proof Object

Component: `src/gpuwrf/physics/noahmp/snow.py::noahmp_snow`
Oracle: verbatim pristine-WRF Noah-MP snow routines compiled with gfortran (fp64
via `-fdefault-real-8`), transcribed unchanged from
`/home/enric/src/wrf_pristine/WRF/phys/module_sf_noahmplsm.F`
(SNOWWATER/SNOWFALL/COMPACT/COMBINE/DIVIDE/COMBO/SNOWH2O/SNOW_AGE/SNOWALB_CLASS).

## Gate: pristine-WRF savepoint parity

`proofs/noahmp/snow_parity.py` -> `snow_savepoint_parity.json`

14 single-column scenarios spanning the full snow life-cycle:

| # | scenario | WRF ISNOW | got ISNOW | result |
|---|---|---|---|---|
| 1 | zero-snow no-op (common Canary case) | 0 | 0 | PASS |
| 2 | light snowfall -> shallow+first-layer+DIVIDE | -2 | -2 | PASS |
| 3 | heavier snowfall -> 3 layers | -3 | -3 | PASS |
| 4 | existing layer + snowfall + compaction + DIVIDE | -3 | -3 | PASS |
| 5 | deep single layer -> DIVIDE to 3 | -3 | -3 | PASS |
| 6 | 2 layers, melt phase change (IMELT), QRAIN | -3 | -3 | PASS |
| 7 | 3 layers mixed, QRAIN percolation | -3 | -3 | PASS |
| 8 | thin top layer -> COMBINE collapse | -2 | -2 | PASS |
| 9 | shallow sublimation to zero | 0 | 0 | PASS* |
| 10 | single-layer heavy sublimation -> COMBINE | 0 | 0 | PASS* |
| 11 | frost onto shallow snow | 0 | 0 | PASS* |
| 12 | warm fresh snowfall -> CLASS albedo refresh (cosz>0) | -3 | -3 | PASS |
| 13 | nighttime aging only (cosz<0) | -2 | -2 | PASS |
| 14 | two thin layers -> phase-1 collapse | -1 | -1 | PASS |

`all_pass = true`. Worst absolute field diff over all scenarios (fp64 round-off):

| field | worst |abs diff| |
|---|---|
| ISNOW | 0 (exact) |
| SNOWH | 1.1e-16 |
| SNEQV | 2.8e-14 |
| TAUSS (snow age) | 3.2e-17 |
| ALBOLD (CLASS albedo) | 0 (exact) |
| SNICE | 2.8e-14 |
| SNLIQ | 1.8e-15 |
| TSNO (snow-layer T) | 0 (exact) |
| ZSNSO | 4.4e-16 |
| soil top-layer SH2O | 0 (exact) |

SWE-budget conservation residual = 0 for every layered scenario.

\* Scenarios 9/10/11 exercise sublimation/frost (QSNSUB/QSNFRO). Those inputs are
produced by the energy/water sprint and are NOT in the frozen `noahmp_snow`
signature, so they are verified through the internal SNOWWATER column path
(`_snowwater_column`, which accepts qsnsub/qsnfro) — identical physics. The public
`noahmp_snow` handles 1-8/12-14 through the frozen signature.

## Gate: vectorized branch-free kernel + zero-snow degrade

`proofs/noahmp/snow_unit.py`

- Mixed (2,3) grid with {zero-snow, shallow, single, 2-layer, 3-layer} columns
  advanced in ONE jitted call -> multiple ISNOW states coexist (no python
  layer-count branching; mask + `where`).
- Zero-snow columns with zero snowfall return SNEQV/SNOWH/ISNOW/SNICE/SNLIQ
  bit-clean unchanged (the common Canary case).
- All snow fields finite (no NaN/Inf from masked empty layers).
- `UNIT PASS`.

## Reproduce

```
# build oracle (once)
conda run -n wrfbuild gfortran -O2 -fdefault-real-8 -fdefault-double-8 \
  -ffree-line-length-none proofs/noahmp/oracle/snow_oracle.f90 \
  -o proofs/noahmp/oracle/snow_oracle
proofs/noahmp/oracle/snow_oracle proofs/noahmp/fixtures/snow_oracle_savepoints.txt
# parity + unit (CPU, fp64)
JAX_PLATFORM_NAME=cpu python proofs/noahmp/snow_parity.py
JAX_PLATFORM_NAME=cpu python proofs/noahmp/snow_unit.py
```
