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

## Comprehensive Diagnostic Harness (2026-05-28)

`test_diagnostic_harness.py` is the canonical entry point for the
per-operator + per-step diagnostic harness. Source lives at
`src/gpuwrf/diagnostics/comprehensive_harness.py`; design at
`.agent/sprints/2026-05-28-diagnostic-harness/design.md`.

The two M9 PLACEHOLDER tests
(`test_physics_couplers_PLACEHOLDER.py`,
`test_operational_variables_PLACEHOLDER.py`) now invoke the harness in slim
form and assert that operator verdicts and operational-variable blocks
populate correctly. They skip cleanly in CPU-only environments because
`State.zeros` requires a JAX GPU device.

### Production driver

To run the full diagnostic harness and emit
`proofs/diagnostic_harness/diagnostic_report.json`:

```
taskset -c 0-3 \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.10 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python scripts/run_diagnostic_harness.py --hours 24 --jax-platform cuda
```

For an overhead-measured smoke run (`--measure-overhead` runs the loop twice
with `diagnostic_on=False` then `True` and reports the wall-clock ratio):

```
taskset -c 0-3 \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.10 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python scripts/run_diagnostic_harness.py --hours 1 --measure-overhead --jax-platform cuda
```

### What the report tells you

A human reader opens `diagnostic_report.json` top-to-bottom:

1. `headline_diagnosis` — one paragraph summarizing the most actionable
   anomaly.
2. `first_failure_attribution` — three pointers: first invariant break,
   first nonfinite cell, first significant wrfout RMSE divergence.
3. `operator_attribution_24h` — per-operator verdict (`ACTIVE`,
   `NOISY_ZERO`, `MISSING`, `INACTIVE`, `PASSIVE_OK`) with per-field
   mean/max Δ statistics.
4. `internal_consistency_24h` — per-invariant first violation step +
   operator + total count.
5. `wrf_anchor_comparison` — hourly wrfout RMSE per field (reused from
   `proofs/m9/operational_trace_hourly.json` if present).
6. `coupling_chain_audit` — pairwise upstream→downstream chain verdicts
   (e.g. `surface_layer__to__mynn_theta_bottom_bc`).
7. `next_sprint_recommendations` — actionable bullet list.

The harness adds operators only via wrapper imports — it does NOT modify
`src/gpuwrf/dycore/**` or `src/gpuwrf/coupling/physics_couplers.py`. The
`diagnostic_on: bool` static-arg flag on `run_diagnostic_forecast` makes the
instrumentation tree dead-code-eliminate when False, so production calls
pay zero overhead.
