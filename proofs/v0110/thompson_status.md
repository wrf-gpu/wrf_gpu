# v0.11.0 Thompson Parity Debts

## Status

- Validated code commit: `52f4e23cc0814e7957543bc2dbb34709e1664780`.
- `cloud_water_sedimentation`: closed in the current base and revalidated. The WRF cloud-water sedimentation path is active by default. P1-5 isolation shows `qc_mean_rel` improves from `0.00712012462600741` with cloud-water sedimentation disabled to `7.533631778533897e-06` with it enabled.
- `snow_fall_speed`: closed in this sprint. Snow terminal velocity now uses the WRF Field two-gamma moment-ratio formulation from `module_mp_thompson.F:3711-3721` instead of the prior single-slope `av_s*xDs**bv_s` approximation. P1-5 `qs_mean_rel` improves from `0.08875921588640973` to `0.054308734078313554`, within the frozen `0.15` mean-relative band.
- `invalid_column_fallback`: quantified and carried. Removing the fallback made `tests/test_m6b_20260509_microphysics_fix.py::test_20260509_bad_cell_invalid_pressure_does_not_create_cloud_feedback` fail with large cloud feedback. The checked WRF column inputs do not exercise the fallback: P1-5 has `0/352` invalid cells, v090 active-precip has `0/228228`, and the d02 sanity has `0/461736`.
- `NSED_MAX`: unchanged at the validated cap `16`.

## Proofs

- `proofs/v0110/thompson_parity.json`: aggregate debt proof. Overall pass is `true`; closed debts are cloud-water sedimentation and snow fall speed; carried debt is invalid-column fallback.
- `proofs/p1_5/precip_water_parity.json`: load-bearing WRF precipitating-column oracle. All gates pass: surface precip ratio `0.9985990860792249`, per-column max rel `0.0014834061486995462`, `qc_mean_rel=7.533631778533897e-06`, `qs_mean_rel=0.054308734078313554`, water closure rel `2.370207011902978e-06`.
- `proofs/v0110/thompson_d02_sanity.json`: one real d02 Thompson step over the full initial-condition columns. GPU was used only because the real d02 `State` loader requires a GPU. Result: finite, zero invalid fallback activation, zero precip/cloud-water sink, and zero water-budget residual.
- `proofs/v0110/thompson_v090_savepoint.json`: secondary active-precip savepoint check. It is not the closure gate here: `qs` is inactive/zero in this savepoint, and the remaining false fields are the pre-existing warm-rain `qc` strict-band miss and theta fp32-storage tolerance artifact.

## Commands

- `taskset -c 0-27 env PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false python proofs/p1_5/precip_water_parity.py`
- `taskset -c 0-27 env PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false python proofs/v090/thompson_savepoint_parity.py --out proofs/v0110/thompson_v090_savepoint.json`
- `/tmp/wrf_gpu_run.sh taskset -c 0-27 env PYTHONPATH=src GPUWRF_CANAIRY_ROOT=<DATA_ROOT>/canairy_meteo OMP_NUM_THREADS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 TF_GPU_ALLOCATOR=cuda_malloc_async python -`
- `taskset -c 0-27 env PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false pytest -q tests/test_m6b_20260509_microphysics_fix.py::test_20260509_bad_cell_invalid_pressure_does_not_create_cloud_feedback tests/test_m5_thompson_column_shapes.py::test_debug_false_hlo_has_no_debug_assert_ops`

## Test Notes

- The bad-pressure invalid-cell regression passes with the fallback retained.
- `tests/test_m5_thompson_column_shapes.py::test_debug_false_hlo_has_no_debug_assert_ops` still fails: production HLO has one extra fusion versus the hand-stripped sibling because the production source retains the invalid-column fallback and the stripped sibling does not. Fixing that cleanly requires either updating the stripped sibling or redesigning the invalid-input guard, both outside this lane's `thompson_column.py`-only ownership.
- The P1-5 proof reports a graupel `qg` residual (`mean_rel=0.6130569422857403`, `max_rel=1.0`) but does not gate it. That residual is not one of the scoped P1-5 debts in this lane and is now called out explicitly in `thompson_parity.json`.

## Next Decision

If the manager requires closing the invalid-column fallback debt now, route it to an Opus second-line debug lane with permission to touch the debug-stripped sibling and/or the upstream invalid-pressure state construction. Within this lane, the WRF parity and d02 sanity inputs show zero operational activation of the fallback.
