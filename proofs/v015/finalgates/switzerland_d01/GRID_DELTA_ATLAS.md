# V0.14 Grid-Delta Atlas

Generated UTC: `2026-06-13T11:01:11.368984+00:00`

## Verdict

- verdict: `FAIL_TOLERANCE`
- paired wrfout files: `72`
- compared numeric fields: `102`
- tolerance manifest supplied: `True`
- plots: `ok`

This report is produced by offline tooling from existing CPU-WRF and GPU wrfout files. It does not run model code.

## Artifacts

- manifest: `<DATA_ROOT>/wrf_gpu_validation/v015_switzerland_d01_72h_gpu_finalgates_20260613T094842Z/grid_delta_atlas/manifest.json`
- summary: `<DATA_ROOT>/wrf_gpu_validation/v015_switzerland_d01_72h_gpu_finalgates_20260613T094842Z/grid_delta_atlas/grid_delta_summary.json`
- assets: `<DATA_ROOT>/wrf_gpu_validation/v015_switzerland_d01_72h_gpu_finalgates_20260613T094842Z/grid_delta_atlas_assets`

## Coverage

- cases: `1`
- domains: `d01`
- lead hours: `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72]`
- variable union/common: `366`/`103`
- CPU-only/GPU-only variables: `259`/`4`

## Inventory Issues

- missing records: `18936`
- mandatory-core missing records: `0`
- non-numeric records: `72`
- nonfinite records: `0`
- shape mismatches: `0`
- dimension-name mismatches: `0`

## Top Field Differences

| Field | RMSE | Bias | p50 | p95 | p99 | p99.9 | Max abs | Corr | Worst lead |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `QNRAIN` | 2.130e+06 | 2.402e+05 | 0 | 234.549 | 9.141e+06 | 3.109e+07 | 8.860e+07 | 0.027 | 45 |
| `QNICE` | 6.033e+04 | 5564.472 | 0 | 1.019e+05 | 2.794e+05 | 6.436e+05 | 2.081e+06 | 0.097 | 1 |
| `PBLH` | 212.465 | -91.847 | 87.598 | 476.252 | 698.420 | 971.675 | 1451.925 | 0.689 | 37 |
| `SWNORM` | 169.945 | 91.047 | 0 | 392.851 | 422.672 | 450.549 | 473.006 | NA | 60 |
| `SWDOWN` | 86.153 | 35.288 | 0 | 240.963 | 326.560 | 376.631 | 407.714 | 0.843 | 12 |
| `SWDNB` | 86.153 | 35.288 | 0 | 240.963 | 326.560 | 376.631 | 407.714 | 0.843 | 12 |
| `PH` | 25.220 | 4.156 | 11.653 | 52.450 | 83.762 | 140.173 | 403.668 | 1.000 | 11 |
| `HFX` | 27.689 | 3.361 | 12.329 | 58.575 | 94.149 | 150.660 | 379.371 | 0.771 | 3 |
| `SWUPT` | 83.648 | -39.291 | 0 | 223.893 | 276.251 | 313.616 | 346.312 | 0.913 | 12 |
| `SWUPB` | 30.696 | -3.890 | 0 | 55.150 | 160.334 | 226.835 | 279.419 | 0.760 | 60 |
| `MU` | 42.566 | -14.974 | 24.612 | 88.847 | 119.399 | 155.657 | 264.259 | 0.999 | 10 |
| `LH` | 21.802 | -10.550 | 7.932 | 46.100 | 74.538 | 106.079 | 249.605 | 0.504 | 33 |
| `P` | 15.205 | -0.621 | 2.193 | 36.029 | 59.225 | 93.340 | 208.305 | 1.000 | 9 |
| `LWDNB` | 51.895 | -35.897 | 42.641 | 85.408 | 94.847 | 103.345 | 108.448 | 0.597 | 42 |
| `GLW` | 51.895 | -35.897 | 42.641 | 85.408 | 94.847 | 103.345 | 108.448 | 0.597 | 42 |
| `OLR` | 26.779 | 15.064 | 10.341 | 60.037 | 76.107 | 85.036 | 102.525 | 0.269 | 72 |
| `LWUPT` | 26.779 | 15.064 | 10.341 | 60.037 | 76.107 | 85.036 | 102.525 | 0.269 | 72 |
| `QKE` | 0.422 | -0.019 | 3.743e-05 | 0.633 | 1.979 | 4.611 | 43.225 | 0.926 | 21 |
| `SNOWNC` | 2.985 | -0.608 | 0.010 | 5.848 | 13.506 | 27.201 | 41.436 | 0.715 | 70 |
| `LWUPB` | 3.022 | -2.025 | 2.057 | 5.828 | 9.740 | 12.565 | 37.750 | 0.997 | 72 |

## Lead-Time Stability

| Field | RMSE slope/h | Bias slope/h | Late-early RMSE | Late-early bias | Worst lead |
| --- | ---: | ---: | ---: | ---: | ---: |
| `QNRAIN` | -1570.805 | -1387.994 | 3.307e+05 | 1.471e+04 | 20 |
| `QNICE` | -970.449 | 117.173 | -3.818e+04 | 2220.863 | 4 |
| `PBLH` | 1.378 | -0.105 | 78.954 | -26.821 | 38 |
| `SWDNB` | -0.495 | -0.527 | -21.520 | -20.206 | 11 |
| `SWDOWN` | -0.495 | -0.527 | -21.520 | -20.206 | 11 |
| `SWUPT` | -0.331 | 0.365 | -16.267 | 16.074 | 11 |
| `OLR` | -0.167 | -0.194 | -4.656 | -3.850 | 2 |
| `LWUPT` | -0.167 | -0.194 | -4.656 | -3.850 | 2 |
| `LH` | -0.251 | 0.027 | -9.354 | 2.169 | 13 |
| `SWUPB` | 0.079 | -0.153 | -0.314 | -3.827 | 59 |
| `SWNORM` | -0.108 | -0.108 | -17.159 | -17.199 | 60 |
| `HFX` | -0.111 | -0.080 | -4.144 | -1.962 | 60 |

## Plot Inventory

- `heatmap`: `<DATA_ROOT>/wrf_gpu_validation/v015_switzerland_d01_72h_gpu_finalgates_20260613T094842Z/grid_delta_atlas_assets/heatmap_rmse.png`
- `heatmap`: `<DATA_ROOT>/wrf_gpu_validation/v015_switzerland_d01_72h_gpu_finalgates_20260613T094842Z/grid_delta_atlas_assets/heatmap_bias.png`
- `heatmap`: `<DATA_ROOT>/wrf_gpu_validation/v015_switzerland_d01_72h_gpu_finalgates_20260613T094842Z/grid_delta_atlas_assets/heatmap_p99_abs.png`
- `heatmap`: `<DATA_ROOT>/wrf_gpu_validation/v015_switzerland_d01_72h_gpu_finalgates_20260613T094842Z/grid_delta_atlas_assets/heatmap_max_abs.png`
- `timeseries`: `<DATA_ROOT>/wrf_gpu_validation/v015_switzerland_d01_72h_gpu_finalgates_20260613T094842Z/grid_delta_atlas_assets/core_fields_rmse_timeseries.png`
- `dashboard`: `<DATA_ROOT>/wrf_gpu_validation/v015_switzerland_d01_72h_gpu_finalgates_20260613T094842Z/grid_delta_atlas_assets/dashboard.png`
- `spatial_max_abs`: `<DATA_ROOT>/wrf_gpu_validation/v015_switzerland_d01_72h_gpu_finalgates_20260613T094842Z/grid_delta_atlas_assets/spatial_maxabs_PH.png`
- `spatial_max_abs`: `<DATA_ROOT>/wrf_gpu_validation/v015_switzerland_d01_72h_gpu_finalgates_20260613T094842Z/grid_delta_atlas_assets/spatial_maxabs_MU.png`
- `spatial_max_abs`: `<DATA_ROOT>/wrf_gpu_validation/v015_switzerland_d01_72h_gpu_finalgates_20260613T094842Z/grid_delta_atlas_assets/spatial_maxabs_P.png`
- `spatial_max_abs`: `<DATA_ROOT>/wrf_gpu_validation/v015_switzerland_d01_72h_gpu_finalgates_20260613T094842Z/grid_delta_atlas_assets/spatial_maxabs_RAINNC.png`
