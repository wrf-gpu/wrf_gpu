# V0.14 Moist-CQW GPU h1-h4 Validation

Verdict: `MOIST_CQW_GPU_H4_ACCEPT`

- run root: `/mnt/data/wrf_gpu_validation/v014_canary_d02_moistcqw_h4_20260610T165255Z`
- previous PSFC-fix baseline: `/mnt/data/wrf_gpu_validation/v014_canary_d02_psfcfix_h4_20260610T160708Z`
- GPU rc: `0`
- compare JSON: `/mnt/data/wrf_gpu_validation/v014_canary_d02_moistcqw_h4_20260610T165255Z/canary_d02_h4_grid_compare.json`
- resource CSV: `/mnt/data/wrf_gpu_validation/v014_canary_d02_moistcqw_h4_20260610T165255Z/resources/v014_canary_d02_moistcqw_h4_gpu_usage.csv`
- peak VRAM: `16921 MiB`; avg GPU util: `38.4%`; max power: `281.8 W`

## Field RMSE Delta

Negative delta means the moist-cqw run improved over the previous PSFC-fix h1-h4 baseline.

| Field | Old RMSE | New RMSE | Delta | Old bias | New bias |
| --- | ---: | ---: | ---: | ---: | ---: |
| `PSFC` | 45.775480 | 41.012950 | -4.762530 | 33.503934 | 38.358605 |
| `MU` | 45.666265 | 45.284746 | -0.381519 | 37.774006 | 42.566177 |
| `P` | 55.124835 | 22.642088 | -32.482748 | -27.408215 | 8.398484 |
| `PH` | 44.729084 | 42.495175 | -2.233909 | 6.533043 | -10.415878 |
| `W` | 0.028076 | 0.024980 | -0.003097 | 0.001212 | 0.001320 |
| `T` | 0.310350 | 0.255805 | -0.054545 | 0.003756 | -0.003011 |
| `QVAPOR` | 0.000379 | 0.000380 | 0.000001 | -0.000012 | -0.000009 |
| `U` | 0.505111 | 0.383930 | -0.121181 | 0.174868 | 0.210567 |
| `V` | 0.441904 | 0.272432 | -0.169473 | 0.013834 | -0.029130 |
| `U10` | 0.695101 | 0.482210 | -0.212891 | -0.314950 | 0.060809 |
| `V10` | 0.860632 | 0.519033 | -0.341599 | 0.387363 | 0.018589 |
| `HFX` | 42.288303 | 40.423357 | -1.864946 | 11.144321 | 10.219274 |
| `LH` | 22.983552 | 20.353543 | -2.630009 | -3.526054 | -1.177740 |
| `PBLH` | 89.286201 | 74.383382 | -14.902819 | -2.303685 | -10.670574 |
| `GLW` | 3.882067 | 3.856382 | -0.025684 | -0.541124 | 0.431188 |
| `SWDOWN` | 27.825622 | 27.867652 | 0.042030 | -13.907487 | -13.928433 |
| `SWNORM` | 28.699370 | 28.740171 | 0.040801 | -13.899573 | -13.920518 |

## `P+PB(k0)` Hydrostatic Residual

Means/RMSEs compare the lowest mass-level total pressure against the run's own dry or moist hydrostatic half-level column. The fix should move GPU pressure away from the old dry-balanced residual and toward the CPU/WRF moist-column regime.

| Source | Mean vs moist Pa | RMSE vs moist Pa | Mean vs dry Pa | RMSE vs dry Pa |
| --- | ---: | ---: | ---: | ---: |
| `cpu_truth` | -13.349 | 13.444 | 186.428 | 188.994 |
| `old_gpu_psfcfix` | -201.492 | 204.437 | -5.990 | 9.268 |
| `new_gpu_moistcqw` | -9.492 | 11.758 | 186.049 | 189.241 |

## Manager Decision

The short GPU gate is accepted if the run is finite/green, pressure-state fields improve materially, and no wind/temperature/moisture regression appears. The all-field comparator may still report `FAIL` from stricter static/base-state or surface-flux lanes; that is not by itself a rejection of this moist-cqw dynamics sprint.
