# Tester Report

## Tests Added Or Run
This sprint changed release governance and launched validation, not source code.
The checks were therefore operational/documentation checks:

- `python scripts/close_sprint.py .agent/sprints/2026-06-10-v014-field-parity-release-gate`
  was run once before reports were filled and correctly failed on stub reports.
- NetCDF inspection confirmed Switzerland 24h `wrfbdy_d01` has eight times only:
  `2023-01-15_00:00:00` through `2023-01-15_21:00:00`.
- Filesystem inspection found no `wrfrst_d0*` in the Switzerland 24h roots.
- Canary inventory found 15 complete L2 d02 CPU-WRF 72h cases with 73 frames.
- Canary L3 d03 retained truth is currently 24h-oriented.

## Results
The release-gate change is technically justified:

- Switzerland 24h resume is rejected.
- Canary d02 is selected for the mandatory 72h gate because complete CPU truth
  exists and it matches the current live-nested validation path.
- The corrected Switzerland 72h CPU baseline launcher is alive and currently in
  the GFS download/build phase.

## Fixtures Used
- `/mnt/data/wrf_gpu_validation/v014_switzerland_cpu24_20260610T073414Z/run_cpu`
- `/mnt/data/wrf_gpu_switzerland_128/run_cpu`
- `/mnt/data/wrf_gpu_switzerland_big/run_cpu`
- `/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output`
- `/mnt/data/canairy_meteo/runs/wrf_l3`

## Gaps
No final CPU-WRF 72h Switzerland truth exists yet from this sprint; it is
running. No GPU 72h validation was launched because the short GPU falsifier is
still occupying the GPU and exact-branch memory preflight should be rerun before
long GPU gates.

## Decision
Decision:

Proceed. Documentation and launch path are correct; keep monitoring the running
CPU baseline and do not treat TOST as a release blocker.
