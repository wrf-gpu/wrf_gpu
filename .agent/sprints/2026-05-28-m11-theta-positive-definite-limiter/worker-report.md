# Worker Report - M11 theta positive-definite limiter

## objective

Replace the operational theta `[200K,450K]` envelope fallback with a positive-definite theta increment limiter, add INV-10 per-step limiter diagnostics, rerun Canary 20260521 24h, and emit the contracted M11 proofs.

## files changed

- `src/gpuwrf/runtime/operational_mode.py`
  - Removed `_limit_theta_by_level`.
  - Added dry-mass-weighted positive-definite theta increment limiter.
  - Added per-level origin-state monotonic bounds and finite positive dry-mass guard split.
  - Added INV-10 diagnostic scan path via `run_forecast_operational_with_limiter_diagnostics`.
- `tests/savepoint/test_dycore_limiter.py`
  - Added focused limiter tests for bounded positivity, first limited cell, mass residual, no-op diagnostics, and monotonic bounds.
- `proofs/m11/limiter_diagnostics_24h.json`
- `proofs/m11/post_m11_skill_diff.json`
- `proofs/m11/divergence_map_v3.json`
- Additional source proof artifacts in `proofs/m11/`: pipeline run, wrfout inventory, speedup, station status, and raw hourly trace.

## commands run

- `python -m py_compile src/gpuwrf/runtime/operational_mode.py tests/savepoint/test_dycore_limiter.py`
- `taskset -c 0-3 pytest -q tests/savepoint/test_dycore_limiter.py`
- `taskset -c 0-3 pytest -q tests/savepoint/test_dycore_100_steps.py`
- `taskset -c 0-3 env PYTHONPATH=src JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 python scripts/m7_daily_pipeline.py --run-id 20260521_18z_l3_24h_20260522T133443Z --hours 24 --output-dir /tmp/m11_theta_pd_limiter_20260521 --proof-dir proofs/m11 --run-root /mnt/data/canairy_meteo/runs/wrf_l3 --domain d02`
- `taskset -c 0-3 env PYTHONPATH=src python scripts/m7_gpu_vs_cpu_skill_diff.py --gpu-root /tmp/m11_theta_pd_limiter_20260521 --cpu-run /mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z --output proofs/m11/post_m11_skill_diff.json --variables T2 U10 V10`
- `taskset -c 0-3 env PYTHONPATH=src python scripts/operational_trace_compare.py --case 20260521 --domain d02 --gpu-pipeline-run proofs/m11/pipeline_run_20260521.json --gpu-root /tmp/m11_theta_pd_limiter_20260521 --wrf-root /mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z --hours 24 --output proofs/m11/operational_trace_hourly_v3.json`
- `taskset -c 0-3 env PYTHONPATH=src JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 python - <<'PY' ...` to generate `proofs/m11/limiter_diagnostics_24h.json`.

## proof objects produced

- `proofs/m11/limiter_diagnostics_24h.json`
  - Status: `PASS`
  - Total step logs: 8,640
  - Limited step count: 8,640
  - First limited cell: `[73, 27, 2]` at global step 1
  - Max theta clip count in one step: 315,351
  - Max absolute theta mass residual: 0.02734375
- `proofs/m11/divergence_map_v3.json`
  - Theta 24h RMSE: 7.50392988504 K
  - M9 v2 theta 24h RMSE baseline: 77.3589529727313 K
  - Headline theta RMSE reduction: 90.29985593563411 %
  - Trace status remains `FAIL` due residual downstream field divergence.
- `proofs/m11/post_m11_skill_diff.json`
  - Verdict: `FAIL_SKILL_DIFF`
  - GPU RMSE: T2 13.111946791392588, U10 7.618270919252392, V10 7.614133182228181.
- `proofs/m11/pipeline_run_20260521.json`
  - Verdict: `PIPELINE_PARTIAL`
  - 24 wrfouts written, finite state check passed, wrfout inventory passed, speedup passed.

## unresolved risks

- AC4 is not fully satisfied. Theta RMSE reduced by more than 30 %, but T2 and U10 station RMSE are worse than the M10 GPU baseline:
  - M10 T2 10.80126250539068 -> M11 T2 13.111946791392588
  - M10 U10 7.237992941570328 -> M11 U10 7.618270919252392
  - M10 V10 7.621687554701543 -> M11 V10 7.614133182228181
- Pipeline verdict remains `PIPELINE_PARTIAL` because the M7 daily runner does not run station scoring unless `--score` is requested; the separate skill diff proof was generated successfully.
- The limiter is strongly active on every timestep. That stabilized theta and improved trace RMSE, but the high clip counts indicate the dycore still produces large out-of-bounds increments that should be diagnosed rather than treated as solved.
- The divergence map remains wrfout-level evidence, not per-operator WRF fixture evidence.

## next decision needed

Decide whether to accept this M11 limiter as the new guard foundation despite station-skill regression, or dispatch a follow-up to reduce limiter aggressiveness / diagnose the dycore increments before moving to the next physics milestone.

## verdict

M11_PARTIAL - headline theta RMSE reduction: 90.29985593563411 %.
