# Tester Report

Decision: ACCEPT_WITH_NARROWER_BLOCKER.

The implementation passes the required CPU proof/test gates and does not claim
full Step-1 parity. It proves that the prior thermodynamic input boundary is no
longer the active blocker.

## Commands Run

- `python -m py_compile` on changed production, test, and proof files.
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src pytest -q tests/test_v014_dry_source_leaf_wiring.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_thermo_column_inputs.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_tsk_znt_sourcing_fix.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_source_fidelity_closure.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/mynn_driver_source_output_fix.py`
- `python -m json.tool` on all four proof JSONs.
- `git diff --check`.

## Results

- Focused test: `2 passed`.
- Primary proof verdict: `THERMO_COLUMN_INPUTS_FIXED_NEXT_BLOCKER_SURFACE_LAYER_OUTPUTS`.
- TSK/ZNT proof verdict: `TSK_ZNT_THERMO_INPUTS_FIXED_NEXT_BLOCKER_SURFACE_LAYER_OUTPUTS`.
- Source-fidelity verdict: `STEP1_SOURCE_FIDELITY_NOT_CLOSED_NARROW_BLOCKER_SFCLAY_OUTPUT_ALGEBRA`.

## Residual Test Gap

No GPU run was attempted. That is intentional: the active issue remains a
CPU/WRF-hooked Step-1 correctness boundary, and long validation is paused until
the field divergence is fixed or bounded.
