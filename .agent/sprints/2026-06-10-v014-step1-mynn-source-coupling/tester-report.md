Decision: PASS for sprint acceptance as a strict narrowing, not as Step-1 closure.

Commands run by worker and rerun by manager:

`python -m py_compile proofs/v014/step1_mynn_source_coupling.py proofs/v014/step1_sfclay_output_algebra.py proofs/v014/step1_source_fidelity_closure.py proofs/v014/mynn_driver_source_output_fix.py src/gpuwrf/coupling/physics_couplers.py src/gpuwrf/runtime/operational_mode.py`

`JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src pytest -q tests/test_v014_mynn_surface_layer_regressions.py tests/test_m6_surface_layer_kernel.py tests/test_v014_dry_source_leaf_wiring.py tests/test_v014_mynn_coldstart_init.py`

Result: `13 passed, 1 skipped`.

Proof reruns:
`proofs/v014/step1_mynn_source_coupling.py`, `proofs/v014/step1_sfclay_output_algebra.py`, `proofs/v014/step1_source_fidelity_closure.py`, and `proofs/v014/mynn_driver_source_output_fix.py` all completed on CPU. JSON validation passed for all four proof outputs. `git diff --check` passed.

Remaining test risk:
The strict Step-1 field gate is still red. The acceptance here is the contract's acceptable narrowing path: MYNN raw source units are bounded with WRF inputs, and the next blocker is identified as the surface/land heat-moisture flux handoff before MYNN.
