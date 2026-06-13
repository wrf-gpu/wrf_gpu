# V0.14 Grid-Delta Atlas

Generated UTC: `2026-06-13T13:19:23.323523+00:00`

## Verdict

- verdict: `FAIL_TOLERANCE`
- paired wrfout files: `72`
- compared numeric fields: `102`
- tolerance manifest supplied: `True`
- plots: `ok`

This report is produced by offline tooling from existing CPU-WRF and GPU wrfout files. It does not run model code.

## Artifacts

- manifest: `/mnt/data/wrf_gpu_validation/v015_canary_d02_72h_gpu_finalgates_20260613T095113Z/grid_delta_atlas/manifest.json`
- summary: `/mnt/data/wrf_gpu_validation/v015_canary_d02_72h_gpu_finalgates_20260613T095113Z/grid_delta_atlas/grid_delta_summary.json`
- assets: `/mnt/data/wrf_gpu_validation/v015_canary_d02_72h_gpu_finalgates_20260613T095113Z/grid_delta_atlas_assets`

## Coverage

- cases: `1`
- domains: `d02`
- lead hours: `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72]`
- variable union/common: `379`/`103`
- CPU-only/GPU-only variables: `272`/`4`

## Inventory Issues

- missing records: `19872`
- mandatory-core missing records: `0`
- non-numeric records: `72`
- nonfinite records: `0`
- shape mismatches: `0`
- dimension-name mismatches: `0`

## Top Field Differences

| Field | RMSE | Bias | p50 | p95 | p99 | p99.9 | Max abs | Corr | Worst lead |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `QNICE` | 540.042 | -5.348 | 0 | 0 | 0 | 0.013 | 5.298e+05 | NA | 30 |
| `QNRAIN` | 8.419 | -0.088 | 0 | 0 | 0 | 0.144 | 3122.397 | -5.881e-06 | 20 |
| `LH` | 85.074 | -15.288 | 28.604 | 66.292 | 467.931 | 928.867 | 1785.424 | -0.024 | 68 |
| `HFX` | 51.817 | 12.607 | 7.781 | 19.309 | 287.932 | 616.987 | 996.897 | 0.897 | 21 |
| `SWNORM` | 42.316 | 2.890 | 8.182 | 59.932 | 95.883 | 473.390 | 951.435 | 0.994 | 19 |
| `SWDNB` | 41.447 | 2.771 | 8.608 | 59.133 | 70.214 | 472.686 | 951.435 | 0.995 | 19 |
| `SWDOWN` | 27.135 | 2.342 | 0.397 | 11.357 | 31.281 | 484.107 | 947.279 | 0.998 | 19 |
| `PBLH` | 105.522 | -25.976 | 59.199 | 220.854 | 324.526 | 429.950 | 818.368 | 0.620 | 19 |
| `SWUPT` | 18.999 | -1.725 | 0.651 | 12.742 | 29.219 | 331.485 | 600.697 | 0.950 | 19 |
| `PH` | 58.685 | -1.527 | 24.270 | 126.718 | 225.733 | 330.401 | 463.467 | 1.000 | 60 |
| `P` | 33.559 | -12.341 | 21.729 | 70.161 | 101.859 | 133.435 | 226.823 | 0.999 | 21 |
| `MU` | 54.546 | 29.673 | 38.132 | 109.493 | 137.865 | 169.763 | 217.488 | 0.954 | 60 |
| `SWUPB` | 5.208 | 0.373 | 0.733 | 4.933 | 12.121 | 80.946 | 181.829 | 0.991 | 19 |
| `LWDNB` | 10.267 | -1.571 | 5.025 | 17.882 | 32.186 | 89.279 | 113.483 | 0.608 | 20 |
| `GLW` | 10.267 | -1.571 | 5.025 | 17.882 | 32.186 | 89.275 | 113.482 | 0.608 | 20 |
| `LWUPB` | 3.767 | -0.024 | 0.109 | 3.217 | 19.273 | 43.288 | 108.778 | 0.979 | 20 |
| `OLR` | 7.803 | 3.774 | 2.787 | 20.490 | 29.671 | 41.568 | 83.615 | 0.516 | 30 |
| `LWUPT` | 7.803 | 3.774 | 2.787 | 20.490 | 29.671 | 41.568 | 83.615 | 0.516 | 30 |
| `SWDNT` | 40.889 | 1.531 | 10.687 | 75.778 | 77.844 | 78.249 | 78.364 | 0.997 | 13 |
| `TSK` | 0.527 | 2.716e-04 | 0 | 0.471 | 2.127 | 7.085 | 18.282 | 0.986 | 20 |

## Lead-Time Stability

| Field | RMSE slope/h | Bias slope/h | Late-early RMSE | Late-early bias | Worst lead |
| --- | ---: | ---: | ---: | ---: | ---: |
| `PBLH` | -0.516 | 1.788 | -25.752 | 67.960 | 12 |
| `QNICE` | -1.650 | 0.071 | -250.098 | 10.695 | 31 |
| `MU` | 0.463 | 0.719 | 12.223 | 23.755 | 60 |
| `LH` | 0.856 | -0.011 | 31.600 | 0.763 | 68 |
| `PH` | 0.336 | 0.316 | 13.722 | 22.249 | 60 |
| `PSFC` | 0.037 | 0.472 | 1.681 | 18.010 | 18 |
| `HFX` | 0.403 | 0.054 | 14.871 | 1.191 | 20 |
| `SWDNT` | 0.226 | -0.064 | 13.953 | 5.100 | 13 |
| `SWDOWN` | -0.164 | -0.070 | -9.207 | -3.535 | 21 |
| `SWNORM` | 0.057 | -0.136 | 4.369 | -0.173 | 21 |
| `SWDNB` | 0.044 | -0.138 | 3.659 | -0.284 | 21 |
| `SWUPT` | -0.126 | 0.045 | -6.556 | 3.062 | 21 |

## Plot Inventory

- `heatmap`: `/mnt/data/wrf_gpu_validation/v015_canary_d02_72h_gpu_finalgates_20260613T095113Z/grid_delta_atlas_assets/heatmap_rmse.png`
- `heatmap`: `/mnt/data/wrf_gpu_validation/v015_canary_d02_72h_gpu_finalgates_20260613T095113Z/grid_delta_atlas_assets/heatmap_bias.png`
- `heatmap`: `/mnt/data/wrf_gpu_validation/v015_canary_d02_72h_gpu_finalgates_20260613T095113Z/grid_delta_atlas_assets/heatmap_p99_abs.png`
- `heatmap`: `/mnt/data/wrf_gpu_validation/v015_canary_d02_72h_gpu_finalgates_20260613T095113Z/grid_delta_atlas_assets/heatmap_max_abs.png`
- `timeseries`: `/mnt/data/wrf_gpu_validation/v015_canary_d02_72h_gpu_finalgates_20260613T095113Z/grid_delta_atlas_assets/core_fields_rmse_timeseries.png`
- `dashboard`: `/mnt/data/wrf_gpu_validation/v015_canary_d02_72h_gpu_finalgates_20260613T095113Z/grid_delta_atlas_assets/dashboard.png`
- `spatial_max_abs`: `/mnt/data/wrf_gpu_validation/v015_canary_d02_72h_gpu_finalgates_20260613T095113Z/grid_delta_atlas_assets/spatial_maxabs_PH.png`
- `spatial_max_abs`: `/mnt/data/wrf_gpu_validation/v015_canary_d02_72h_gpu_finalgates_20260613T095113Z/grid_delta_atlas_assets/spatial_maxabs_P.png`
- `spatial_max_abs`: `/mnt/data/wrf_gpu_validation/v015_canary_d02_72h_gpu_finalgates_20260613T095113Z/grid_delta_atlas_assets/spatial_maxabs_MU.png`
- `spatial_max_abs`: `/mnt/data/wrf_gpu_validation/v015_canary_d02_72h_gpu_finalgates_20260613T095113Z/grid_delta_atlas_assets/spatial_maxabs_QVAPOR.png`
