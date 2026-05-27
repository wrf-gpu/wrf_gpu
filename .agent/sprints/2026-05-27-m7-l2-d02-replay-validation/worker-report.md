# Worker Report — M7 L2 d02 Replay Validation

Summary: Implemented the L2 d02 replay adapter and orchestrator without dycore, physics, runtime, validation-core, or writer changes. The 24h GPU forecast completed on the latest complete L2 run (`20260525_18z_l2_72h_20260526T221207Z`) using L2 `d01` parent-history boundary forcing interpolated to child `d02` side strips. Verdict: `L2_D02_BOUNDED_FAIL`. Bounds and wall-clock proofs passed, but Tier-4 RMSE missed all contracted surface thresholds, so nested-grid backfills are not publish-ready from this evidence.

## Objective

Validate the operational L2 one-way nested d02 path by loading L2 d02 initial state, deriving d02 boundary leaves from L2 d01 hourly wrfouts, running a 24h GPU forecast, and comparing output against L2 d02 Gen2 reference wrfouts.

## Files Changed

- `src/gpuwrf/integration/d02_replay.py`
- `scripts/m7_l2_d02_replay.py`
- `tests/test_m7_l2_d02_replay.py`
- `.agent/sprints/2026-05-27-m7-l2-d02-replay-validation/l2_inventory.json`
- `.agent/sprints/2026-05-27-m7-l2-d02-replay-validation/tier4_rmse_l2_d02.json`
- `.agent/sprints/2026-05-27-m7-l2-d02-replay-validation/bounds_check_l2_d02.json`
- `.agent/sprints/2026-05-27-m7-l2-d02-replay-validation/wall_clock_l2_d02.json`
- `.agent/sprints/2026-05-27-m7-l2-d02-replay-validation/l2_d02_validation_summary.json`
- `.agent/sprints/2026-05-27-m7-l2-d02-replay-validation/pipeline_run_l2_d02.json`
- `.agent/sprints/2026-05-27-m7-l2-d02-replay-validation/pipeline_run_20260521.json`
- `.agent/sprints/2026-05-27-m7-l2-d02-replay-validation/wrfout_inventory.json`
- `.agent/sprints/2026-05-27-m7-l2-d02-replay-validation/station_scores_20260521.json`
- `.agent/sprints/2026-05-27-m7-l2-d02-replay-validation/speedup_vs_cpu_24h.json`
- `.agent/sprints/2026-05-27-m7-l2-d02-replay-validation/restart_in_pipeline.json`
- `.agent/sprints/2026-05-27-m7-l2-d02-replay-validation/repeatability.json`
- `.agent/sprints/2026-05-27-m7-l2-d02-replay-validation/worker-report.md`

Large wrfout outputs were written only under `/tmp/m7_pipeline_runs/l2_d02_20260525_18z_l2_72h_20260526T221207Z/` and are not committed.

## Commands Run

- `python -m py_compile src/gpuwrf/integration/d02_replay.py scripts/m7_l2_d02_replay.py tests/test_m7_l2_d02_replay.py`
  - Output: no output; exit 0.
- `pytest -q tests/test_m7_l2_d02_replay.py`
  - Output: `.... [100%]` and `4 passed in 1.11s`; exit 0.
- `taskset -c 0-3 python scripts/m7_l2_d02_replay.py --inventory-only`
  - Output: `{"inventory": "/tmp/wrf_gpu2_l2d02/.agent/sprints/2026-05-27-m7-l2-d02-replay-validation/l2_inventory.json", "run_count": 28}`; exit 0.
- `taskset -c 0-3 python scripts/m7_l2_d02_replay.py --hours 24`
  - Output summary: verdict `L2_D02_BOUNDED_FAIL`, run `20260525_18z_l2_72h_20260526T221207Z`, statuses `pipeline=PIPELINE_PARTIAL`, `rmse=FAIL`, `bounds=PASS`, `wall_clock=PASS`; exit 2 because the orchestrator returns nonzero for bounded fail.
- `python scripts/validate_agentos.py`
  - Output: `{"errors": [], "ok": true, "required_files_checked": 31, "skills_checked": 13}`; exit 0.
- `git diff --check`
  - Output: no output; exit 0.

## Proof Objects

- Inventory: `.agent/sprints/2026-05-27-m7-l2-d02-replay-validation/l2_inventory.json`
  - 28 L2 run directories inventoried; 3 full d01+d02 72h runs found.
- RMSE: `.agent/sprints/2026-05-27-m7-l2-d02-replay-validation/tier4_rmse_l2_d02.json`
  - `FAIL`: T2 RMSE `4.0673 K` > `3 K`; U10 RMSE `10.7825 m s-1` > `7.5`; V10 RMSE `7.8268 m s-1` > `7.5`.
- Bounds: `.agent/sprints/2026-05-27-m7-l2-d02-replay-validation/bounds_check_l2_d02.json`
  - `PASS`: theta lower-30 range `290.795..355.362 K`, full theta range `290.795..492.672 K`, max winds U/V/W `71.046/51.940/0.654 m s-1`.
- Wall clock: `.agent/sprints/2026-05-27-m7-l2-d02-replay-validation/wall_clock_l2_d02.json`
  - `PASS`: total `280.647 s`, forecast-only `245.383 s`, device `cuda:0`, CPU affinity `[0,1,2,3]`.
- Summary: `.agent/sprints/2026-05-27-m7-l2-d02-replay-validation/l2_d02_validation_summary.json`
  - Publish-readiness conclusion: not publish-ready for nested-grid backfills from this sprint evidence.

## Risks

- The parent-to-child boundary adapter uses bilinear interpolation on WRF parent index coordinates. It is a data-loading adapter, not a WRF nest interpolation parity proof; the RMSE miss may be from interpolation fidelity, the operational model, or both.
- `PIPELINE_PARTIAL` is expected from the reused daily pipeline when station scoring is not requested; the decisive failure is the contracted Tier-4 RMSE proof, not station scoring.
- Contract hard rule says no remote push and local commit only, while the generic launch text requested push. I followed the sprint contract and did not remote-push.

## Handoff

Objective handled end-to-end with proof artifacts. Next decision needed: whether to improve parent-to-child boundary interpolation/parity, rerun against a different complete L2 day, or classify L2 nested backfills as blocked until WRF nest-boundary interpolation is matched more closely.
