# Worker Report - M11.3 Coordinated Dycore Fix

Verdict: `M11P3_PARTIAL`.

## objective

Apply the coordinated agy dycore fixes for restored operational RK advection, acoustic physical `mu`, and `theta_1` decoupling; then verify with the 1h diagnostic harness, 24h limiter diagnostic, 24h daily pipeline, and the known-tautological 100-step savepoint sanity test.

## files changed

- `src/gpuwrf/runtime/operational_mode.py`
  - `_rk_scan_step.advance_stage` now calls `compute_advection_tendencies` before the acoustic substep and adds horizontal pressure-gradient tendencies on top.
  - WRF reference documented in code: `dyn_em/module_em.F:rk_scalar_tend`.
  - Added a minimal operational wrapper consistency fix so `acoustic.mu` (`advanced["mu"]`) persists as the physical perturbation across acoustic substeps.
- `src/gpuwrf/dynamics/core/acoustic.py`
  - `_decouple_theta_after_advance` now uses `state.theta_1` in the numerator.
  - `acoustic_substep_core` now returns `mu=advanced["mu"]`.
  - Pressure diagnosis now uses `advanced["mu"]` on the same physical perturbation basis.
- `proofs/m11p3/diagnostic_report_after_fix.json`
- `proofs/m11p3/limiter_diagnostics_24h.json`
- `proofs/m11p3/pipeline_run_20260521.json`
- `proofs/m11p3/post_m11p3_skill_diff.json`
- `proofs/m11p3/dycore_100_steps_pytest.txt`
- Pipeline sidecar proofs in `proofs/m11p3/`: `wrfout_inventory.json`, `station_scores_20260521.json`, `speedup_vs_cpu_24h.json`, `restart_in_pipeline.json`, `repeatability.json`.

## commands run

- `python -m py_compile src/gpuwrf/runtime/operational_mode.py src/gpuwrf/dynamics/core/acoustic.py`
- `taskset -c 0-3 env PYTHONPATH=src JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 python scripts/run_diagnostic_harness.py --hours 1 --radiation-cadence-steps 999999 --jax-platform cuda --output proofs/m11p3/diagnostic_report_after_fix.json`
- `taskset -c 0-3 env PYTHONPATH=src JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 python - <<'PY' ...` to generate `proofs/m11p3/limiter_diagnostics_24h.json`.
- `taskset -c 0-3 env PYTHONPATH=src JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 python scripts/m7_daily_pipeline.py --run-id 20260521_18z_l3_24h_20260522T133443Z --hours 24 --output-dir /tmp/m11p3_coordinated_dycore_20260521 --proof-dir proofs/m11p3 --run-root /mnt/data/canairy_meteo/runs/wrf_l3 --domain d02 --score`
- `taskset -c 0-3 env PYTHONPATH=src python scripts/m7_gpu_vs_cpu_skill_diff.py --gpu-root /tmp/m11p3_coordinated_dycore_20260521 --cpu-run /mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z --output proofs/m11p3/post_m11p3_skill_diff.json --variables T2 U10 V10`
- `taskset -c 0-3 pytest -q tests/savepoint/test_dycore_100_steps.py`

## proof objects produced

- `proofs/m11p3/diagnostic_report_after_fix.json`
  - `dycore_rk3`: `ACTIVE`.
  - Failed AC4: first invariant break is `theta_in_bounds` at step 11 after `dycore_rk3`.
  - Failed AC4: `wind_in_bounds` also first violates at step 11.
  - Failed AC4: first nonfinite is step 12 after `dycore_rk3` with 27,072 cells.
  - Harness wall time: 101.68 s.
- `proofs/m11p3/limiter_diagnostics_24h.json`
  - Failed AC5: limited step count `8640/8640`; limiter drop is `0%`, not the target `>=90%`.
  - Failed AC5: max cells limited per step `461,736`, worse than the M11 315k reference and far above the 10% target.
  - Failed AC5: max absolute theta mass residual is `Infinity`, not bounded `<=0.05`.
  - Diagnostic wall time: 291.51 s.
- `proofs/m11p3/pipeline_run_20260521.json`
  - Failed AC6: `PIPELINE_BLOCKED` after forecast hour 1.
  - No wrfouts produced.
  - All-finite check failed; nonfinite fields include `qke`, `fltv`, `qv_flux`, `theta_flux`, `tau_u`, `tau_v`, `ustar`, and extreme dycore state magnitudes.
  - T2 RMSE not measurable.
- `proofs/m11p3/post_m11p3_skill_diff.json`
  - `BLOCKED_NO_GPU_WRFOUTS`; skill diff could not run because the pipeline produced no `wrfout_d02_*` files.
- `proofs/m11p3/speedup_vs_cpu_24h.json`
  - `BLOCKED`; no 24h speedup number is valid because the pipeline blocked at hour 1.
- `proofs/m11p3/dycore_100_steps_pytest.txt`
  - Passed: `1 passed in 428.38s`.
  - Limited value only: per agy, this test is a JAX-vs-JAX self-compare tautology and does not prove WRF correctness.

## unresolved risks

- The coordinated fix did not stabilize the operational dycore. Restored advection plus the acoustic physical-`mu`/`theta_1` changes still produce dycore blow-up before 1h.
- The physical-`mu` basis exposed additional consistency requirements in operational pressure/mass wrapping. I patched the minimal local inconsistencies found, but the acceptance gates still fail.
- No GPU performance or speedup claim is supported. The only measured timings are failed/diagnostic timings, not a successful 24h forecast.
- T2/U10/V10 skill cannot be assessed because no wrfouts were produced.

## next decision needed

Dispatch a narrower dycore failure localization sprint around the first 12 steps after restoring advection, with per-RK-stage and per-acoustic-substep state probes for `mu`, `muts`, pressure, theta, and winds. Do not advance to downstream physics milestones on this state.

## verdict

`M11P3_PARTIAL` - limiter drop `0%`; 24h pipeline `PIPELINE_BLOCKED`; T2 RMSE unavailable.
