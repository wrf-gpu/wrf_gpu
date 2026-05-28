# M17 Worker Report

## Verdict

`M17_PARTIAL`

Headline: Thompson is not silently failed over the contracted 1h diagnostic horizon; T2 RMSE reduction is `-0.008%` versus the 10.801 K post-iter2 baseline, so the >=20% target was not met.

## Objective

Root-cause the Thompson microphysics NOISY_ZERO smoke finding, apply the minimal legitimate fix in the Thompson adapter if a coupling bug exists, and re-verify with the diagnostic harness, 100-step parity, and Canary 20260521 24h skill diff.

## Files Changed

- `.agent/sprints/2026-05-28-m17-microphysics-silent-failure-fix/root_cause_analysis.md`
- `.agent/sprints/2026-05-28-m17-microphysics-silent-failure-fix/worker-report.md`
- `proofs/m17/diagnostic_report_before_fix_no_radiation.json`
- `proofs/m17/diagnostic_report_after_fix.json`
- `proofs/m17/thompson_initial_condition_probe.json`
- `proofs/m17/post_m17_skill_diff.json`

No model code was changed. The evidence did not support a Thompson adapter bug.

## Commands Run

- `taskset -c 0-3 env PYTHONPATH=src JAX_ENABLE_X64=true JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 python scripts/run_diagnostic_harness.py --hours 1 --output proofs/m17/diagnostic_report_before_fix.json`
  - Failed before report: RRTMG autotune OOM while allocating ~922 MiB.
- `taskset -c 0-3 env PYTHONPATH=src JAX_ENABLE_X64=true JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false TF_GPU_ALLOCATOR=cuda_malloc_async OMP_NUM_THREADS=4 python scripts/run_diagnostic_harness.py --hours 1 --radiation-cadence-steps 999999 --output proofs/m17/diagnostic_report_before_fix_no_radiation.json`
  - Passed; Thompson already `ACTIVE`.
- `taskset -c 0-3 env PYTHONPATH=src JAX_ENABLE_X64=true JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false python -u - <<PY ...`
  - Produced `proofs/m17/thompson_initial_condition_probe.json`.
- `taskset -c 0-3 env PYTHONPATH=src JAX_ENABLE_X64=true JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false TF_GPU_ALLOCATOR=cuda_malloc_async OMP_NUM_THREADS=4 python scripts/run_diagnostic_harness.py --hours 1 --radiation-cadence-steps 999999 --output proofs/m17/diagnostic_report_after_fix.json`
  - Passed; Thompson `ACTIVE`.
- `taskset -c 0-3 env PYTHONPATH=src JAX_ENABLE_X64=true JAX_PLATFORMS=cuda JAX_PLATFORM_NAME=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 pytest -q tests/savepoint/test_dycore_100_steps.py`
  - Passed: `1 passed in 1580.29s (0:26:20)`.
- `taskset -c 0-3 env PYTHONPATH=src JAX_ENABLE_X64=true JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu OMP_NUM_THREADS=4 python scripts/m7_gpu_vs_cpu_skill_diff.py --gpu-root /tmp/m12_surface_flux_mynn_20260521 --output proofs/m17/post_m17_skill_diff.json`
  - Produced `FAIL_SKILL_DIFF`; 24 common valid times, 73 stations.

## Proof Objects Produced

- `proofs/m17/thompson_initial_condition_probe.json`
  - Initial 20260521 `qc/qr/qi/qs/qg/Ni/Nr` are all zero.
  - Full-domain qv is subsaturated everywhere: max liquid supersaturation `-0.0723501209000571`.
- `proofs/m17/diagnostic_report_after_fix.json`
  - `microphysics_thompson = ACTIVE`.
  - Nonzero mean deltas for `qv/qc/qr/qi/qs/qg/theta`.
  - `surface_layer = ACTIVE`, `mynn_pbl = ACTIVE`, `lateral_boundary = ACTIVE`, `boundary_guards = PASSIVE_OK`.
  - `rrtmg = INACTIVE` because radiation cadence was disabled to avoid the known RRTMG autotune OOM.
- `proofs/m17/post_m17_skill_diff.json`
  - T2 GPU RMSE `10.80208347450218 K`.
  - Baseline post-iter2 T2 GPU RMSE `10.80126250539068 K`.
  - Reduction `-0.008%`; target was `>=20%`.
- `tests/savepoint/test_dycore_100_steps.py`
  - Passed.

## Acceptance Status

- AC1 root cause documented: `DONE`.
- AC2 Thompson adapter fix: `NOT APPLIED`; no adapter bug was found, and a fake source term would be unphysical.
- AC3 1h diagnostic harness: `PASS` for Thompson activity, with radiation disabled. It also reports `theta_in_bounds` first violates at dycore step 141, outside M17 writable scope.
- AC4 100-step parity: `PASS`.
- AC5 24h skill diff: `FAIL`; no T2 improvement. The skill diff uses the latest complete 20260521 24h wrfouts (`/tmp/m12_surface_flux_mynn_20260521`) because the current full-radiation path OOMs in RRTMG autotuning and no Thompson behavior changed.
- AC6 worker report: `DONE`.

## Unresolved Risks

- The 3-step diagnostic harness can false-positive `NOISY_ZERO` for physics schemes when the initial state is physically inactive for that scheme. For Thompson on 20260521, the 1h horizon is the reliable proof.
- The 1h harness exposes a separate dycore/theta problem: `theta_in_bounds` first violates at step 141 with `first_violation_operator = dycore_rk3`.
- Full-radiation 1h/24h reruns are blocked by RRTMG XLA autotune OOM in this worktree. The no-radiation harness isolates Thompson but does not validate RRTMG.

## Next Decision Needed

Do not spend another Thompson-adapter sprint on this smoke finding. Dispatch the next fix to the dycore/theta-bound violation or RRTMG memory/autotune path, depending on whether correctness or full-radiation proof throughput is the immediate priority.
