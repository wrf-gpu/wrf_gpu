# Worker Report

Summary: source-fidelity sprint reduced the Step-1 gap to one MYNN
driver/kernel source-output blocker.

## Objective

Close, or reduce to one strictly narrower WRF-anchored blocker, the Step-1
`T_TENDF` source-fidelity gap.

## Files Changed

- `src/gpuwrf/coupling/physics_couplers.py`
- `src/gpuwrf/runtime/operational_mode.py`
- `tests/test_v014_dry_source_leaf_wiring.py`
- `proofs/v014/step1_source_fidelity_closure.py`
- `proofs/v014/step1_source_fidelity_closure.json`
- `proofs/v014/step1_source_fidelity_closure.md`
- `.agent/reviews/2026-06-10-v014-step1-source-fidelity-closure.md`
- refreshed reused Step-1 proof artifacts.

## Outcome

Verdict:

`STEP1_SOURCE_FIDELITY_NOT_CLOSED_NARROW_BLOCKER_MYNN_DRIVER_SOURCE_OUTPUT`

The sprint implemented `rqvblten` exposure and WRF
`conv_t_tendf_to_moist` handling in the `rad_rk_tendf=1` source path. That
removed radiation/moist-conversion as primary blockers. The strict Step-1
proof remains red because JAX MYNN source outputs are too weak.

## Commands Run

- `python -m py_compile ...`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src pytest -q tests/test_v014_dry_source_leaf_wiring.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_source_fidelity_closure.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_dry_source_leaf_fix.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_part2_source_leaves_split.py`
- `python -m json.tool ...`
- `git diff --check`

## Next Boundary

Emit one WRF MYNN driver hook at Step 1 around `module_bl_mynnedmf_driver` and
compare exact MYNN inputs/outputs against JAX `_mynn_column_from_state` and
`step_mynn_pbl_column`.
