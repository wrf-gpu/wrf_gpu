# V0.14 PSFC Moist Pressure GPU h1-h4 Validation

Date: 2026-06-10

## Verdict

`PSFC_DIAGNOSTIC_GPU_H4_CONFIRMED`

The WRF moist-hydrostatic `PSFC` diagnostic fix from commit `a08553dc` was
validated in a short Canary d02 GPU run. The old vapor-light `PSFC` floor is
gone in live GPU output. The remaining h1-h4 `PSFC` residual tracks the
dry-mass / pressure-state lanes rather than the missing vapor-column load.

## Run

- run root:
  `/mnt/data/wrf_gpu_validation/v014_canary_d02_psfcfix_h4_20260610T160708Z`
- branch/head: `worker/gpt/v013-close-manager`, `a08553dc`
- case: `20260501_18z_l2_72h_20260519T173026Z`
- GPU command: `proofs/v0120/powered_tost_n15/run_one_case_v0120.py --hours 4`
- GPU rc: `0`
- CPU comparator rc: `0`
- paired domain/leads: `d02`, h1-h4
- resource CSV:
  `resources/v014_canary_d02_psfcfix_h4_gpu_usage.csv`
- GPU resource summary: 93 samples, peak `15507 MiB`, average utilization
  `77.7%`, max power `283.0 W`

## Key Field Results

| Field | Overall RMSE | Overall Bias | Worst Lead RMSE | Lead h1 RMSE/Bias | Lead h4 RMSE/Bias |
|---|---:|---:|---:|---:|---:|
| `PSFC` | 45.775 Pa | +33.504 Pa | 57.823 Pa | 57.823 / +51.553 Pa | 35.487 / +18.270 Pa |
| `MU` | 45.666 Pa | +37.774 Pa | 58.079 Pa | 58.079 / +52.861 Pa | 35.153 / +26.430 Pa |
| `P` | 55.125 Pa | -27.408 Pa | 60.383 Pa | 47.942 / -23.756 Pa | 60.383 / -29.168 Pa |
| `PH` | 44.729 | +6.533 | 52.817 | 41.161 / +16.974 | 52.817 / -9.489 |
| `QVAPOR` | 3.787e-4 | -1.206e-5 | 5.129e-4 | n/a | n/a |

The previous fixed-LBC old-formula h1/h4 values were `PSFC` RMSE
`156.974 Pa` / `186.741 Pa`; after the fix they are `57.823 Pa` / `35.487 Pa`.

## Remaining Blocker

The h1-h4 comparator still reports `FAIL`, but `PSFC` is no longer the dominant
failure. The remaining release blocker is the 3D pressure-state dynamics lane:
operational acoustic w-equation still uses dry `cqw` / `pg_buoy_w_dry`, so
`P/PH/W` remain dry-balanced relative to WRF. This requires the next
moist-cqw dynamics sprint before Switzerland GPU or field-parity promotion.
