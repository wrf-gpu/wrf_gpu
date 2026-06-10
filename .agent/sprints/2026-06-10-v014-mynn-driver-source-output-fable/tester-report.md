# Tester Report

Decision: mechanical gates pass; strict Step-1 scientific gate remains red.

## Manager-Rerun Commands

- `python -m py_compile proofs/v014/mynn_driver_source_output_fix.py proofs/v014/step1_source_fidelity_closure.py proofs/v014/step1_dry_source_leaf_fix.py proofs/v014/step1_part2_source_leaves_split.py proofs/v014/same_input_contract_builder.py tests/test_v014_mynn_coldstart_init.py tests/test_v014_dry_source_leaf_wiring.py src/gpuwrf/physics/mynn_pbl.py src/gpuwrf/coupling/physics_couplers.py src/gpuwrf/integration/d02_replay.py src/gpuwrf/runtime/operational_mode.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src pytest -q tests/test_v014_mynn_coldstart_init.py tests/test_v014_dry_source_leaf_wiring.py tests/test_m5_mynn_column_shapes.py tests/test_mynn_edmf_oracle.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src pytest -q tests/test_v014_mynn_coldstart_init.py tests/test_v014_dry_source_leaf_wiring.py tests/test_m5_mynn_column_shapes.py tests/test_mynn_edmf_oracle.py tests/test_m5_mynn_radicand.py tests/test_v0110_qke_finiteness.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/mynn_driver_source_output_fix.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_source_fidelity_closure.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_dry_source_leaf_fix.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_part2_source_leaves_split.py`
- `python -m json.tool` on all four proof JSONs
- `git diff --check`

## Result

All manager-rerun mechanical gates passed. Final scientific verdict:

`MYNN_SOURCE_ROOT_CAUSED_INIT_QKE_FIXED_KERNEL_PROVEN_NEXT_SFCLAY_STEP1_FLUX_BOUNDARY`
