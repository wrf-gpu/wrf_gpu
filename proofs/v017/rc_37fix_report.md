# v0.17 RC #37 Conditional State Leaves Report

Branch/worktree: `worker/gpt/v017-rc-37fix` in `<USER_HOME>/src/wrf_gpu2/.wt-v017-rc-37fix`

Base: `worker/perf/v017-rc` at `fa0ca55ce3baf5251239ac9065da670346f9f107`

Implementation commit: `642f1d611b33ec25eeaa0fde72cb04e29b1fa49d`

Final branch HEAD including this proof report: `1d2fe145ea0240ec451f6eb655cf0e89cb6604b7`

Note: the parent repository `.git` was read-only in this sandbox, so the branch was created in an isolated local clone rather than mutating `worker/perf/v017-rc` directly.

## Port Summary

- Ported the v0.18-trunk conditional additive-leaf behavior to the diverged v0.17 RC State layout.
- `_state_field_shapes(grid, mp_physics=...)` now allocates only active static-scheme leaves.
- Default `mp_physics=8` leaves `nwfa`, `nifa`, `qh`, `Nh`, `qvolg`, `qvolh`, and `hail_acc` as `None`.
- `State.zeros(..., mp_physics=...)`, `State.from_init(..., mp_physics=...)`, `State.active_field_names()`, and `State.ensure_conditional_leaves(...)` now express the active leaf contract.
- WSM7/WDM7 scan adapters materialize hail leaves once before use; Thompson-aero materializes aerosol leaves before cold-start/adapter use.
- Operational precision enforcement, checkpoint, restart, wrfrst, debug snapshots, sharding, and boundary feedback now respect active fields and skip inactive conditional `None` leaves.
- wrfrst default mp=8 output omits both exact `GPUWRF_STATE_*` conditional variables and optional WRF variables `QNWFA`, `QNIFA`, `QHAIL`, `QNHAIL`, `QVGRAUPEL`, `QVHAIL`, and `HAILNC`.

## Default Leaf Count Proof

CPU proof command reported:

```text
mp=8  shapes=60 active=60 leaves=60 conditional=() expected=()
mp=24 shapes=65 active=65 leaves=65 conditional=('qh', 'Nh', 'qvolg', 'qvolh', 'hail_acc') expected=('qh', 'Nh', 'qvolg', 'qvolh', 'hail_acc')
mp=26 shapes=65 active=65 leaves=65 conditional=('qh', 'Nh', 'qvolg', 'qvolh', 'hail_acc') expected=('qh', 'Nh', 'qvolg', 'qvolh', 'hail_acc')
mp=28 shapes=62 active=62 leaves=62 conditional=('nwfa', 'nifa') expected=('nwfa', 'nifa')
```

Guard test: `tests/test_v017_conditional_state_leaves.py`.

## Test Results

Passed:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= pytest -q \
  tests/test_v017_conditional_state_leaves.py \
  tests/test_v017_qh_hail_state.py \
  tests/test_v016_thompson_aero_threading.py \
  tests/test_v016_thompson_aero_oracle.py \
  tests/test_m6_precision_matrix.py \
  tests/test_m7_restart_checkpoint_roundtrip.py \
  tests/test_p0_5_restart_full_carry.py \
  tests/test_v0110_wrfrst_netcdf.py \
  tests/test_noahmp_checkpoint_v2.py \
  tests/contracts/test_v060_physics_interfaces.py \
  tests/unit/test_decouple_theta_state_reference.py \
  tests/test_v017_gfs_pbl_operational.py \
  tests/test_v017_fp32_physics.py
```

Result: `64 passed, 2 skipped, 9 warnings in 25.55s`.

Passed:

```bash
PYTHONPATH=src python -m py_compile \
  src/gpuwrf/contracts/state.py \
  src/gpuwrf/coupling/boundary_feedback.py \
  src/gpuwrf/coupling/physics_couplers.py \
  src/gpuwrf/coupling/scan_adapters.py \
  src/gpuwrf/runtime/operational_mode.py \
  src/gpuwrf/runtime/checkpoint.py \
  src/gpuwrf/io/restart.py \
  src/gpuwrf/io/wrfrst_netcdf.py \
  src/gpuwrf/debug/snapshots.py \
  src/gpuwrf/runtime/sharding.py
```

Exploratory broader CPU run outside the RC37 surface produced unrelated failures:

- `tests/test_m6b3_scratch_state_parity.py::test_m6b3_synthetic_dryrun_catches_scratch_perturbations` failed because its proof output directory was absent in the isolated clone.
- Three `tests/test_rrtm_lw_operational_wiring.py` cases produced NaNs under the CPU/AOT setup, with XLA CPU machine-feature mismatch warnings.

These were not caused by the conditional State-leaf port and were not modified.

## Unresolved Risks

- No GPU tests were run, per instruction. CPU-only GPU-gated tests were skipped.
- JAX emitted read-only persistent-cache warnings for `<USER_HOME>/.cache/gpuwrf/jit`; tests still passed.
- No RC divergence blocked the port.
