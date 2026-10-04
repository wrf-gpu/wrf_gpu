# v0.23 P-BUNDLE Report

Date: 2026-07-02
Branch: `worker/gpt/v023-perf-bundle`
GPU: not used; no GPU lock taken.

## Commits

- `1d05d764` P6 clean dead code and JAX idioms
- `6a26f145` P2 gate safe floors and finite guard
- `2a3079ab` P4 vectorize calc_coef_w loop1
- `cb65c286` P1 add default-off single scan knob
- `68730ac1` P7b add fused low-effort compile knob

## P6 - Dead Code / Idiom Hygiene

Files changed:

- `src/gpuwrf/dynamics/core/acoustic.py`
- `src/gpuwrf/dynamics/core/__init__.py`
- `src/gpuwrf/dynamics/vertical_implicit_solver.py`
- `src/gpuwrf/dynamics/core/rhs_ph.py`
- `src/gpuwrf/dynamics/core/rk_addtend_dry.py`
- `src/gpuwrf/physics/rrtmg_sw.py`

Changes:

- Removed dead `_mass_couple_theta_before_advance`.
- Removed dead `w_solve_core` and its `thomas_solve_scan` import/export.
- Removed production-dead `R_D`, `CP_D`, `GRAVITY_M_S2` constants from `vertical_implicit_solver.py`; validation-only builder now inlines the same literals.
- Replaced scoped `jnp.pad` edge/zero padding idioms with explicit `jnp.concatenate` slice construction.
- Replaced RRTMG-SW zero/mask allocation idioms with broadcasted scalar zeros.
- Did not touch P6 4.21 donation sites.

Bit-identity / gate:

- Default-path numerical semantics are unchanged by construction: padding replacements concatenate the same edge or zero slices; RRTMG replacements only change zero allocation shape construction; deleted functions/constants are production-dead.
- `python -m compileall -q ...` passed for touched modules.
- `python -m pytest tests/unit/test_decouple_theta_state_reference.py tests/test_v014_rhs_ph_real_case.py tests/test_v014_hypsometric_opt2.py tests/test_m5_rrtmg_transfer_solver.py tests/test_m5_rrtmg_column_shapes.py::test_rrtmg_sw_step_preserves_column_shapes_and_fp64_dtype tests/test_m6b_coftz_theta_fix.py tests/test_m6x_adr023_column_solver.py -q` -> 19 passed.

Measured delta:

- Source cleanup only; no runtime knob/default change.
- Dead target grep after edit found no production callers of removed names.

## P2 - Launch-Count Cuts

Files changed:

- `src/gpuwrf/dynamics/core/advance_w.py`
- `src/gpuwrf/dynamics/core/calc_p_rho.py`
- `src/gpuwrf/dynamics/core/rk_addtend_dry.py`
- `src/gpuwrf/dynamics/core/small_step_finish.py`
- `src/gpuwrf/runtime/finite_state_guard.py`
- `tests/test_v0210_finite_state_guard.py`

Changes:

- Gated the seven remaining safe-floor sites behind existing `_safe_floors()` / `_floor_pos()` behavior.
- Collapsed finite-state guard success-path reductions to one scalar reduction by `ravel` + `concatenate`.
- Did not use `jnp.stack`.

Bit-identity / gate:

- Safe floors remain default-off; default physics path keeps the raw values.
- Finite guard returns the same pass/fail semantics; failure diagnostics still fall back to per-field host checks.
- `python -m pytest tests/test_v0210_finite_state_guard.py tests/test_v0222_nested_wallclock.py::test_finite_guard_batches_jax_success_path tests/test_v015_stream_a_bitwise.py::test_advance_w_thomas_unroll_env_default_inert tests/test_v014_moist_cqw_pressure_dynamics.py tests/test_v014_hypsometric_opt2.py -q` -> 16 passed.

Measured delta:

- Seven target safe-floor sites are now gated.
- Finite guard lowering on the mixed-shape probe changed from 3 reductions / vector result to 1 reduction / scalar result.
- Post-edit source check: `finite_state_guard.py` has 1 `jnp.concatenate` and 0 `jnp.stack` mentions.

## P4 - `calc_coef_w` Loop 1 Only

Files changed:

- `src/gpuwrf/dynamics/acoustic_wrf.py`
- `tests/test_m6b0r_jax_top_row_synthetic.py`

Changes:

- Vectorized only Loop 1 in `calc_coef_w_wrf_coefficients`.
- Left Loop 2 / Thomas forward sweep untouched.

Bit-identity / gate:

- Added a synthetic legacy-loop comparison that runs the pre-edit Loop 1 logic and asserts exact `np.array_equal` for `a`, `alpha`, and `gamma`.
- `python -m pytest tests/test_m6b0r_jax_top_row_synthetic.py tests/test_m6b0r_calc_coef_w_fix.py -q -rsx` -> 2 passed, 4 skipped, 2 xfailed.
- The 4 skips are because M6B0-R HDF5 savepoints are absent in this workspace; only `manifest.json` files exist under the savepoint directories.

Measured delta:

- Synthetic HLO scatter mentions: 175 -> 140.
- Synthetic HLO text size: 33,759 chars -> 29,756 chars.
- No change to the sequential Thomas recurrence.

## P1 - Single-Scan Selection Knob

Files changed:

- `src/gpuwrf/runtime/operational_mode.py`
- `tests/test_v023_single_scan_knob.py`
- `proofs/v023/perf_bundle/P_BUNDLE_QUESTION.md`

Changes:

- Added `GPUWRF_SINGLE_SCAN` selection knob.
- Default remains OFF.

Bit-identity / gate:

- `python -m pytest tests/test_v023_single_scan_knob.py -q` -> 2 passed.
- Equivalence proof is the existing artifact `proofs/perf/segscan_equiv.json`, per manager resolution:
  - status `PASS`
  - `seg_vs_single`: `BITWISE (max abs diff == 0 on all fields)`
  - cases `0.2h` and `0.6h`: `bitwise_seg_eq_single=true`, all `max_abs_diff_seg_vs_single` fields are `0.0`
  - the `0.6h` case includes radiation step 180.

Deferred:

- Live `proofs/perf/single_scan_equiv.py` run is deferred because the referenced run directory is absent in this worktree.
- Default flip is deferred to the GPU phase because it needs warm s/step no-regress confirmation.

## P7b - Fused Low-Effort Cold Compile Wrapper

Files changed:

- `src/gpuwrf/runtime/domain_tree.py`
- `tests/test_aot_executable.py`

Changes:

- Added default-off `GPUWRF_FUSED_CASCADE_LOW_EFFORT=1`.
- When enabled, fused-cascade AOT cheap-key/load/cold-compile miss handling runs under:
  - `jax_exec_time_optimization_effort=-1.0`
  - `jax_memory_fitting_effort=-1.0`
  - `jax_optimization_level=O1`
  - `jax_memory_fitting_level=O1`
- The original JAX config values are restored after the fused AOT path.
- Existing cheap-key exec-env hashing separates default and low-effort executables.

Bit-identity / gate:

- Default-off context test proves ambient config is unchanged.
- Opt-in context test proves `exec_env_hash()` changes under the low-effort profile.
- Cold-miss test proves the low-effort config is active during fused compile serialization and restored after return.
- `python -m pytest tests/test_aot_executable.py -q` -> 33 passed.

Measured CPU A/B:

- Synthetic CPU fused-cascade cold compile, fresh processes, same HLO prefix `8f054f4f4d3f`.
- Default profile: wall `0.862821s`, max RSS `1540960 KB`, cheap key prefix `feff665b3fc1`.
- Low-effort O1 profile: wall `0.868997s`, max RSS `1545768 KB`, cheap key prefix `cd9f856a39ff`.
- CPU result is effectively neutral/slightly slower on this synthetic graph; the lever remains opt-in and default-off. Warm GPU s/step no-regress remains GPU-gated.
- O0/O0 probe was rejected locally: same HLO but cold wall regressed to `2.814405s`.

## Final CPU Gate

Command:

```bash
python -m pytest tests/test_v0210_finite_state_guard.py tests/test_v0222_nested_wallclock.py::test_finite_guard_batches_jax_success_path tests/test_v015_stream_a_bitwise.py::test_advance_w_thomas_unroll_env_default_inert tests/test_v014_moist_cqw_pressure_dynamics.py tests/test_v014_hypsometric_opt2.py tests/unit/test_decouple_theta_state_reference.py tests/test_v014_rhs_ph_real_case.py tests/test_m5_rrtmg_transfer_solver.py tests/test_m5_rrtmg_column_shapes.py::test_rrtmg_sw_step_preserves_column_shapes_and_fp64_dtype tests/test_m6b_coftz_theta_fix.py tests/test_m6x_adr023_column_solver.py tests/test_m6b0r_jax_top_row_synthetic.py tests/test_m6b0r_calc_coef_w_fix.py tests/test_v023_single_scan_knob.py tests/test_aot_executable.py -q -rsx
```

Result: 69 passed, 4 skipped, 2 xfailed in 14.10s.

Known residuals:

- M6B0-R HDF5 savepoints are not present in this workspace, so those savepoint oracle tests skip.
- The live `single_scan_equiv.py` run-dir fixture is not present; equivalence is cited from `proofs/perf/segscan_equiv.json` per manager resolution.
- P7b warm s/step no-regress is GPU-phase deferred.
- `ruff` is not installed in this environment.
