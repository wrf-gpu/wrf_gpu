# V0.14 Grid-Cell Envelope

Generated UTC: `2026-06-08T22:50:02.736380+00:00`

This is a grid-field attribution artifact, not a station-skill result and not an equivalence pass.

## Coverage

- writer operational fields declared: `130`
- spatial wrfout cases: `1`
- aggregate-only cases: `2`

## 20260429_18z_l2_72h_20260524T204451Z

- source: `case_json_aggregate_only`
- limitation: Retained GPU wrfout directory is not available; only stored case JSON aggregates can be used.
- aggregate fields available: `T2, U10, V10`
- T2: RMSE `1.028`, bias `0.128`, p95 `1.997`, max `10.439`
- U10: RMSE `2.301`, bias `-0.625`, p95 `4.604`, max `17.817`
- V10: RMSE `4.184`, bias `-2.484`, p95 `6.978`, max `17.512`
- spatial unavailable writer fields: `127`

## 20260430_18z_l2_72h_20260520T191306Z

- source: `case_json_aggregate_only`
- limitation: Retained GPU wrfout directory is not available; only stored case JSON aggregates can be used.
- aggregate fields available: `T2, U10, V10`
- T2: RMSE `0.989`, bias `-9.794e-04`, p95 `1.618`, max `11.482`
- U10: RMSE `2.409`, bias `-0.773`, p95 `4.793`, max `10.216`
- V10: RMSE `3.053`, bias `-1.875`, p95 `5.415`, max `13.241`
- spatial unavailable writer fields: `127`

## 20260501_18z_l2_72h_20260519T173026Z

- source: `spatial_grid_wrfouts`
- common leads compared: `24` (1-24 h)
- dynamic fields with RMSE envelope: `50`
- static/grid fields audited separately: `48`
- time metadata fields audited: `2`
- writer fields not emitted in retained GPU wrfouts: `26`
- emitted writer fields missing in CPU truth: `4`
- incompatible fields: `0`

### Minimum Dynamic Fields

| field | count | bias | RMSE | MAE | p95 abs | p99 abs | max abs | frac tol | r |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `T2` | 251856 | -0.051 | 0.994 | 0.489 | 1.542 | 5.118 | 10.373 | 0.959 | 0.741 |
| `Q2` | 251856 | 8.417e-04 | 0.001 | 0.001 | 0.003 | 0.003 | 0.006 | NA | 0.515 |
| `U10` | 251856 | -0.944 | 2.068 | 1.648 | 4.014 | 5.519 | 11.969 | 0.789 | 0.581 |
| `V10` | 251856 | 1.036 | 2.524 | 1.859 | 5.474 | 8.172 | 16.034 | 0.772 | 0.390 |
| `PSFC` | 251856 | -504.513 | 525.288 | 505.904 | 703.844 | 756.195 | 1.893e+03 | NA | 0.998 |
| `TSK` | 251856 | 0.081 | 3.046 | 0.722 | 6.886 | 16.802 | 31.583 | NA | 0.348 |
| `PBLH` | 251856 | -81.735 | 175.660 | 139.778 | 324.833 | 454.719 | 1.104e+03 | NA | 0.389 |
| `UST` | 251856 | 0.006 | 0.093 | 0.062 | 0.183 | 0.386 | 0.837 | NA | 0.305 |
| `HFX` | 251856 | 1.958 | 64.280 | 17.367 | 84.636 | 380.129 | 688.118 | NA | 0.349 |
| `LH` | 251856 | -13.488 | 36.851 | 29.524 | 70.825 | 103.843 | 283.128 | NA | 0.350 |
| `SWDOWN` | 251856 | -20.078 | 112.944 | 49.845 | 292.913 | 486.270 | 949.428 | NA | 0.960 |
| `GLW` | 251856 | 7.173 | 25.613 | 13.098 | 75.498 | 85.763 | 113.031 | NA | 0.256 |
| `RAINC` | 251856 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA |
| `RAINNC` | 251856 | -0.002 | 0.060 | 0.007 | 0.009 | 0.175 | 4.614 | NA | -0.005 |
| `RAINSH` | 251856 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA |
| `U` | 11151360 | 2.719 | 4.612 | 3.436 | 9.781 | 13.157 | 22.841 | NA | 0.969 |
| `V` | 11249568 | 3.703 | 5.830 | 4.445 | 11.889 | 13.770 | 18.133 | NA | 0.724 |
| `W` | 11333520 | -2.259e-04 | 0.128 | 0.031 | 0.083 | 0.207 | 12.175 | NA | 0.134 |
| `T` | 11081664 | 0.758 | 2.268 | 1.459 | 5.675 | 7.973 | 10.471 | NA | 0.999 |
| `QVAPOR` | 11081664 | -3.379e-08 | 9.499e-04 | 4.853e-04 | 0.002 | 0.003 | 0.007 | NA | 0.972 |
| `P` | 11081664 | -147.639 | 228.122 | 147.867 | 524.177 | 633.364 | 1.286e+03 | NA | 0.993 |
| `PB` | 11081664 | 0.960 | 28.642 | 2.295 | 0.008 | 40.477 | 1.112e+03 | NA | 1.000 |
| `PH` | 11333520 | -244.367 | 336.208 | 251.462 | 651.000 | 777.139 | 926.573 | NA | 0.996 |
| `PHB` | 11333520 | -0.641 | 45.353 | 3.594 | 0.047 | 73.584 | 2.238e+03 | NA | 1.000 |
| `MU` | 251856 | -242.799 | 273.821 | 249.634 | 422.983 | 486.872 | 1.501e+03 | NA | 0.272 |
| `MUB` | 251856 | 3.190 | 58.769 | 7.622 | 1.523 | 268.094 | 1.115e+03 | NA | 1.000 |
| `QCLOUD` | 11081664 | 1.237e-06 | 2.829e-05 | 1.661e-06 | 0 | 0 | 0.001 | NA | -4.773e-04 |
| `QICE` | 11081664 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA |
| `QRAIN` | 11081664 | -2.045e-09 | 3.209e-07 | 9.451e-09 | 0 | 0 | 8.423e-05 | NA | -4.328e-04 |
| `QSNOW` | 11081664 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA |
| `QGRAUP` | 11081664 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA |
| `QNICE` | 11081664 | 0 | 0 | 0 | 0 | 0 | 0 | NA | NA |
| `QNRAIN` | 11081664 | 0.283 | 25.011 | 0.808 | 0 | 0 | 6.995e+03 | NA | -4.308e-04 |
| `QKE` | 11081664 | -0.005 | 0.246 | 0.040 | 0.170 | 0.687 | 10.571 | NA | 0.472 |

### Worst Dynamic Fields

- `PSFC` (surface_diagnostic): RMSE `525.288`, bias `-504.513`, p95 `703.844`, max `1.893e+03`
- `PH` (dynamics_thermodynamics): RMSE `336.208`, bias `-244.367`, p95 `651.000`, max `926.573`
- `MU` (dynamics_thermodynamics): RMSE `273.821`, bias `-242.799`, p95 `422.983`, max `1.501e+03`
- `P` (dynamics_thermodynamics): RMSE `228.122`, bias `-147.639`, p95 `524.177`, max `1.286e+03`
- `PBLH` (surface_diagnostic): RMSE `175.660`, bias `-81.735`, p95 `324.833`, max `1.104e+03`
- `SWNORM` (radiation_flux): RMSE `113.200`, bias `-19.981`, p95 `292.647`, max `949.428`
- `SWDNB` (radiation_flux): RMSE `112.944`, bias `-20.078`, p95 `292.913`, max `949.428`
- `SWDOWN` (surface_diagnostic): RMSE `112.944`, bias `-20.078`, p95 `292.913`, max `949.428`
- `SWUPT` (radiation_flux): RMSE `77.193`, bias `16.134`, p95 `201.660`, max `652.015`
- `HFX` (surface_diagnostic): RMSE `64.280`, bias `1.958`, p95 `84.636`, max `688.118`
- `MUB` (dynamics_thermodynamics): RMSE `58.769`, bias `3.190`, p95 `1.523`, max `1.115e+03`
- `PHB` (dynamics_thermodynamics): RMSE `45.353`, bias `-0.641`, p95 `0.047`, max `2.238e+03`

### Lead Blocks

- `T2`: 0-6h RMSE 1.024 bias -0.025; 6-12h RMSE 1.389 bias 0.121; 12-24h RMSE 0.699 bias -0.150
- `U10`: 0-6h RMSE 1.500 bias -0.443; 6-12h RMSE 1.936 bias -0.365; 12-24h RMSE 2.358 bias -1.485
- `V10`: 0-6h RMSE 1.578 bias -0.771; 6-12h RMSE 3.235 bias 1.979; 12-24h RMSE 2.502 bias 1.469
- `PSFC`: 0-6h RMSE 417.159 bias -380.968; 6-12h RMSE 601.524 bias -585.900; 12-24h RMSE 532.849 bias -525.592
- `U`: 0-6h RMSE 2.109 bias 0.984; 6-12h RMSE 5.013 bias 3.408; 12-24h RMSE 5.268 bias 3.242
- `V`: 0-6h RMSE 2.131 bias -0.519; 6-12h RMSE 5.750 bias 4.322; 12-24h RMSE 7.013 bias 5.505
- `W`: 0-6h RMSE 0.124 bias 0.002; 6-12h RMSE 0.129 bias 0.001; 12-24h RMSE 0.130 bias -0.002
- `T`: 0-6h RMSE 0.689 bias -0.051; 6-12h RMSE 1.499 bias 0.302; 12-24h RMSE 2.988 bias 1.391

### Spatial Splits

- `T2` land/ocean: land RMSE 3.390 bias 1.840; ocean RMSE 0.399 bias -0.200
- `T2` elevation: land_0_300m RMSE 4.064 bias 2.331; land_300_1000m RMSE 2.837 bias 1.498; land_gt_1000m RMSE 2.365 bias 1.277; ocean RMSE 0.399 bias -0.200
- `T2` quadrant: NE RMSE 1.300 bias -0.043; NW RMSE 0.634 bias -0.024; SE RMSE 1.149 bias -0.103; SW RMSE 0.733 bias -0.034
- `U10` land/ocean: land RMSE 2.205 bias -0.420; ocean RMSE 2.057 bias -0.986
- `U10` elevation: land_0_300m RMSE 2.614 bias -0.600; land_300_1000m RMSE 1.936 bias -0.382; land_gt_1000m RMSE 1.438 bias -0.036; ocean RMSE 2.057 bias -0.986
- `U10` quadrant: NE RMSE 2.181 bias -0.932; NW RMSE 2.110 bias -1.315; SE RMSE 1.760 bias -0.471; SW RMSE 2.189 bias -1.059
- `V10` land/ocean: land RMSE 2.242 bias 0.412; ocean RMSE 2.545 bias 1.086
- `V10` elevation: land_0_300m RMSE 2.414 bias 0.471; land_300_1000m RMSE 2.340 bias 0.481; land_gt_1000m RMSE 1.443 bias 0.123; ocean RMSE 2.545 bias 1.086
- `V10` quadrant: NE RMSE 1.586 bias 0.294; NW RMSE 3.036 bias 1.757; SE RMSE 1.877 bias 0.624; SW RMSE 3.197 bias 1.474
- `PSFC` land/ocean: land RMSE 514.606 bias -479.268; ocean RMSE 526.122 bias -506.507
- `PSFC` elevation: land_0_300m RMSE 560.642 bias -514.320; land_300_1000m RMSE 503.418 bias -480.912; land_gt_1000m RMSE 401.877 bias -386.551; ocean RMSE 526.122 bias -506.507
- `PSFC` quadrant: NE RMSE 513.675 bias -487.072; NW RMSE 548.062 bias -530.217; SE RMSE 505.150 bias -487.773; SW RMSE 533.240 bias -513.103
- `U` land/ocean: land RMSE 4.760 bias 3.078; ocean RMSE 4.602 bias 2.695
- `U` elevation: land_0_300m RMSE 5.178 bias 3.197; land_300_1000m RMSE 4.485 bias 2.925; land_gt_1000m RMSE 4.420 bias 3.148; ocean RMSE 4.602 bias 2.695
- `U` quadrant: NE RMSE 5.050 bias 3.005; NW RMSE 4.098 bias 2.260; SE RMSE 4.995 bias 3.111; SW RMSE 4.224 bias 2.501
- `V` land/ocean: land RMSE 5.735 bias 3.544; ocean RMSE 5.837 bias 3.714
- `V` elevation: land_0_300m RMSE 5.195 bias 3.073; land_300_1000m RMSE 6.019 bias 3.811; land_gt_1000m RMSE 6.157 bias 3.944; ocean RMSE 5.837 bias 3.714
- `V` quadrant: NE RMSE 5.002 bias 2.981; NW RMSE 6.233 bias 4.174; SE RMSE 5.387 bias 3.257; SW RMSE 6.569 bias 4.407
- `W` land/ocean: land RMSE 0.425 bias 0.033; ocean RMSE 0.059 bias -0.003
- `W` elevation: land_0_300m RMSE 0.611 bias 0.078; land_300_1000m RMSE 0.130 bias -0.003; land_gt_1000m RMSE 0.141 bias -0.010; ocean RMSE 0.059 bias -0.003
- `W` quadrant: NE RMSE 0.243 bias 0.001; NW RMSE 0.047 bias -8.647e-04; SE RMSE 0.044 bias -0.001; SW RMSE 0.050 bias -4.600e-04
- `SWDOWN` land/ocean: land RMSE 141.937 bias 39.655; ocean RMSE 110.330 bias -24.795
- `SWDOWN` elevation: land_0_300m RMSE 58.448 bias 8.642; land_300_1000m RMSE 154.667 bias 46.399; land_gt_1000m RMSE 234.976 bias 105.175; ocean RMSE 110.330 bias -24.795
- `SWDOWN` quadrant: NE RMSE 101.148 bias -21.779; NW RMSE 98.566 bias -9.359; SE RMSE 145.220 bias -49.764; SW RMSE 100.281 bias 0.353
- `GLW` land/ocean: land RMSE 26.659 bias -3.993; ocean RMSE 25.528 bias 8.055
- `GLW` elevation: land_0_300m RMSE 13.963 bias 3.319; land_300_1000m RMSE 27.203 bias -4.046; land_gt_1000m RMSE 44.354 bias -22.539; ocean RMSE 25.528 bias 8.055
- `GLW` quadrant: NE RMSE 25.046 bias 6.682; NW RMSE 23.751 bias 6.394; SE RMSE 32.879 bias 14.996; SW RMSE 18.862 bias 0.708

### Inventory Exceptions

- missing from retained GPU wrfouts: `QFX, GRDFLX, SNOWC, TSLB, SMOIS, SH2O, SNOW, SNOWH, CANWAT, SFROFF, UDROFF, ALBEDO, EMISS, TSNO, SNICE, SNLIQ, ZSNSO, ISNOW, SNEQVO, CANLIQ, CANICE, ISEEDARR_SPPT, ISEEDARR_SKEBS, ISEEDARRAY_SPP_CONV, ISEEDARRAY_SPP_PBL, ISEEDARRAY_SPP_LSM`
- emitted but missing from CPU truth: `QNSNOW, QNGRAUPEL, QNCLOUD, QNCCN`
- incompatible: `none`
- static/grid mismatch count: `31`
- static mismatch `C2H`: max abs `9.500e+04`, first leads `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`
- static mismatch `C2F`: max abs `9.500e+04`, first leads `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`
- static mismatch `C4F`: max abs `2.678e+04`, first leads `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`
- static mismatch `C4H`: max abs `2.674e+04`, first leads `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`
- static mismatch `HGT`: max abs `228.129`, first leads `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`
- static mismatch `RDN`: max abs `161.674`, first leads `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`
- static mismatch `C1H`: max abs `1.025`, first leads `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`
- static mismatch `C1F`: max abs `1.019`, first leads `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`
- static mismatch `C3F`: max abs `0.282`, first leads `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`
- static mismatch `C3H`: max abs `0.281`, first leads `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`
- static mismatch `XLAT_U`: max abs `0.028`, first leads `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`
- static mismatch `XLAT_V`: max abs `0.027`, first leads `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`

## Ranked Root-Cause Hypotheses

1. **WRF vertical-coordinate / grid-metric payload mismatch is the first root-cause target.**
   Evidence: 31 audited static fields are not exact across checked leads; largest: C2H max 9.500e+04, C2F max 9.500e+04, C4F max 2.678e+04, C4H max 2.674e+04, HGT max 228.129, RDN max 161.674.
   Next probe: Diff DycoreMetrics/GridSpec against CPU wrfinput or CPU wrfout before any timestep, then rerun this envelope after the metric payload is exact.
   Confidence: `high`

2. **Pressure-gradient / mass-wind coupling in the dycore is the leading operator-level suspect.**
   Evidence: 3D wind RMSE U=4.61, V=5.83; PSFC RMSE=525; surface wind RMSE U10=2.07, V10=2.52.
   Next probe: Instrument first-timestep MU/P/PH pressure-gradient tendencies and U/V updates against WRF savepoints.
   Confidence: `high`

3. **Radiation or surface-energy diagnostics are a secondary amplifier of T2/TSK/PBL divergence.**
   Evidence: SWDOWN RMSE=113, GLW RMSE=25.6, COSZEN RMSE=0.0401, T2 RMSE=0.994; T2 bias=-0.05091370028285012.
   Next probe: Keep radiation diagnostics in the next proof gate, but do not fix them before the first-step wind/mass budget is isolated.
   Confidence: `medium`
