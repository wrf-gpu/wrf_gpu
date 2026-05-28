# M12 Worker Report

## Verdict

`M12_PARTIAL`

Headline: post-fix Canary 20260521 24 h T2 RMSE is `10.802 K`; target was `<= 5.0 K`.

## Objective

Fix surface HFX magnitude/sign by replacing the writer fallback with the WRF aerodynamic-resistance formula, audit MYNN bottom-BC sign convention, and test whether the change reduces Canary 20260521 24 h T2/HFX errors.

## Files Changed

- `src/gpuwrf/io/wrfout_writer.py`
- `src/gpuwrf/coupling/physics_couplers.py`
- `.agent/sprints/2026-05-28-m12-surface-flux-mynn-parity/mynn_bottom_bc_audit.md`
- `.agent/sprints/2026-05-28-m12-surface-flux-mynn-parity/worker-report.md`
- `proofs/m12/*`

## What Changed

- HFX/LH writer fallback now recomputes sfclay diagnostics and derives HFX as `rho * cp * (theta_surface - theta_air) / r_a`, with `r_a = FH / (kappa * ustar)`, preferring direct `HFX`/`LH` fields if present.
- MYNN explicit momentum bottom tendency now uses the same `dt/dz * rhosfc/rho0` density scaling as scalar fluxes.
- MYNN bottom-BC sign audit found scalar signs consistent with WRF: positive upward fluxes are added to the bottom RHS.

## Commands Run

- `taskset -c 0-3 pytest -q tests/test_m7_netcdf_writer.py tests/test_m6_surface_layer_kernel.py tests/test_m7_skill_fix_algorithmic.py`
- `taskset -c 0-3 pytest -q tests/savepoint/test_dycore_100_steps.py`
- `taskset -c 0-3 python scripts/m7_daily_pipeline.py --hours 24 --output-dir /tmp/m12_surface_flux_mynn_20260521 --proof-dir proofs/m12`
- `TF_GPU_ALLOCATOR=cuda_malloc_async XLA_PYTHON_CLIENT_PREALLOCATE=false taskset -c 0-3 python scripts/m7_daily_pipeline.py --hours 24 --output-dir /tmp/m12_surface_flux_mynn_20260521 --proof-dir proofs/m12`
- `taskset -c 0-3 python scripts/m7_gpu_vs_cpu_skill_diff.py --gpu-root /tmp/m12_surface_flux_mynn_20260521 --output proofs/m12/post_m12_skill_diff.json`
- `taskset -c 0-3 python - <<PY ... surface_flux_parity_hour_1.json`

## Proof Objects

- `proofs/m12/surface_flux_parity_hour_1.json`: `FAIL`
  - HFX hour-1 RMSE `65.441 W m-2`; only `3.05%` of nontrivial cells within 5%.
  - LH hour-1 RMSE `41.258 W m-2`; only `7.33%` of nontrivial cells within 5%.
- `proofs/m12/post_m12_skill_diff.json`: `FAIL_SKILL_DIFF`
  - T2 RMSE `10.802 K`
  - U10 RMSE `7.231 m s-1`
  - V10 RMSE `7.617 m s-1`
  - HFX 24 h RMSE `978.034 W m-2`; baseline `980.137 W m-2`, drop `0.21%`, target drop `>= 70%`.
- `proofs/m12/pipeline_run_20260521.json`: 24 wrfouts produced, inventory `PASS`, speedup `PASS`, pipeline verdict `PIPELINE_PARTIAL` because station scoring was intentionally left to the skill-diff script.
- `proofs/m12/wrfout_inventory.json`: `PASS`, 24 wrfouts readable with minimum variables present.
- `proofs/m12/speedup_vs_cpu_24h.json`: `PASS`, speedup `14.61x`.
- `.agent/sprints/2026-05-28-m12-surface-flux-mynn-parity/mynn_bottom_bc_audit.md`: sign convention audit.

## Acceptance Status

- AC1 surface flux parity: `FAIL`
- AC2 MYNN bottom-BC sign audit: `PASS`
- AC3 HFX writer fallback formula: `IMPLEMENTED`
- AC4 100-step parity: `PASS`
- AC5 24 h skill gain: `FAIL`
- AC6 worker report: `DONE`

## Unresolved Risks

- The HFX/LH output error is dominated by wrong lower-atmosphere state, not just writer composition. Hour-1 GPU HFX reaches `1047.5 W m-2` while WRF max is `254.3 W m-2`.
- T2 RMSE remains unchanged from the pre-M12 post-iter2 level, so the surface writer fix alone does not address the skill regression.
- The dry MYNN column kernel still computes internal neutral-bulk surface terms before the adapter's explicit real-surface correction; this remains outside the M12 writable scope.

## Next Decision Needed

Prioritize the theta/lower-column state defect before further surface-flux tuning. M12 did not produce the expected skill gain because the atmospheric state feeding sfclay remains physically wrong.
