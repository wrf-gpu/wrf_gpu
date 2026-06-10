# Tester Report

## Tests Added Or Run

The sprint ran CPU-only NetCDF budget probes and inspected the existing h1/h4
grid-comparator outputs from the fixed Canary LBC run.

## Results

The key h1 budget is decisive: CPU `PSFC-(P_TOP+MU+MUB)` mean is `198.734 Pa`,
independent CPU `sum(QVAPOR*dp_dry)` mean is `198.800 Pa`, and GPU has a
physical vapor-column integral near `197.545 Pa`. However, GPU
`PSFC-(P_TOP+MU+MUB)` is `-8.949 Pa`, creating the dominant `~208 Pa` vapor-load
gap.

## Fixtures Used

- `/mnt/data/wrf_gpu_validation/v014_canary_d02_72h_lbcfix_20260610T151455Z/`
- CPU truth:
  `/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z`

## Gaps

No production fix or post-fix GPU validation exists yet. The active Canary 72h
run is useful characterization, not release-green evidence while this residual
stands.

## Decision

Decision:

BLOCK_RELEASE_PROMOTION_PENDING_PSFC_MOIST_PRESSURE_FIX.
