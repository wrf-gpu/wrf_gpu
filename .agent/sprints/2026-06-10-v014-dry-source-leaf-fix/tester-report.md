# Tester Report

Tester: worker and manager rerun

Decision: mechanical gates pass; scientific closure gate remains red.

## Commands

- `python -m py_compile proofs/v014/step1_dry_source_leaf_fix.py proofs/v014/step1_part2_source_leaves_split.py tests/test_v014_dry_source_leaf_wiring.py src/gpuwrf/coupling/physics_couplers.py src/gpuwrf/runtime/operational_mode.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src pytest -q tests/test_v014_dry_source_leaf_wiring.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_dry_source_leaf_fix.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_part2_source_leaves_split.py`
- `python -m json.tool proofs/v014/step1_dry_source_leaf_fix.json`
- `python -m json.tool proofs/v014/step1_part2_source_leaves_split.json`
- `git diff --check`

## Result

All mechanical gates passed. The scientific gate remains red by design:

`DRY_SOURCE_LEAF_PLUMBING_ACTIVE_BUT_STEP1_T_TENDF_NOT_CLOSED`
