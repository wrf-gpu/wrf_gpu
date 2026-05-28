# Savepoint Harness

`tests/savepoint/` is the M8.B scaffold for WRF reference-state comparisons. It keeps the existing M6B6 coupled-step comparator as the first real test surface and gives M9 fixed places to add 1000-step, physics-coupler, and operational-variable references.

The harness runs CPU-only. The comparator writes generated HDF5 savepoints under pytest temporary directories; those binary intermediates are not committed.

## Saved-State Groups

| Group | Producing operator | WRF routine compared against | Current test file | Status |
|---|---|---|---|---|
| `dycore.coupled_step_complete` | `gpuwrf.dynamics.coupled_step.coupled_timesteps_wrf` | `solve_em.F` Runge-Kutta loop plus `small_steps` acoustic loop | `test_dycore_100_steps.py` | Real 100-step column parity wrapper. |
| `dycore.lateral_boundary_replay` | `gpuwrf.coupling.boundary_apply.apply_lateral_boundaries` | `solve_em.F` specified lateral-boundary tendency application | `test_dycore_1000_steps_PLACEHOLDER.py` | `xfail`: M9 will produce reference states. |
| `physics.tendency_couplers` | Thompson, MYNN, RRTMG, and surface-layer adapters | `module_mp_thompson.F`, `module_bl_mynn.F`, `module_ra_rrtmg_*.F`, and WRF surface driver routines | `test_physics_couplers_PLACEHOLDER.py` | `xfail`: M9 will produce reference states. |
| `operational.surface_variables` | `gpuwrf.runtime.operational_mode.run_forecast_operational` | `solve_em.F` operational forecast step plus WRF diagnostic/output variable production | `test_operational_variables_PLACEHOLDER.py` | `xfail`: M9 will produce reference states. |

## Fixtures

`conftest.py` provides:

- `wrf_fortran_reference_paths`: verifies the M6B6 `wrfout` and `wrfbdy` sources exist before real parity runs.
- `wrf_reference_root`: pytest temporary directory for generated reference savepoints.
- `wrf_fortran_reference_loader`: loader for HDF5 savepoint arrays through `gpuwrf.validation.savepoint_io.read_savepoint`.
- `jax_state_under_test_loader`: loader for the current JAX-side coupled-step snapshots.
- `savepoint_groups`: authoritative group metadata for future M9 tests.

## M9 Extension Points

M9 should replace the placeholder bodies with real reference-state tests without changing the existing 100-step guard. New committed artifacts should remain manifests, JSON proof objects, and small metadata only; large WRF or HDF5 payloads stay outside git.
