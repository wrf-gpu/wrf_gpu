# Tester Report

Decision: ACCEPT_WITH_NARROWER_BLOCKER.

The proof and regression gates support the production change as a valid
WRF-sourced fix. The sprint does not close full Step-1 parity, but it proves
that `TSK/ZNT/MAVAIL` are no longer the active source of the remaining
surface-layer divergence.

## Commands Run

- `python -m py_compile proofs/v014/step1_tsk_znt_sourcing_fix.py proofs/v014/step1_sfclay_boundary_fix.py proofs/v014/step1_source_fidelity_closure.py proofs/v014/mynn_driver_source_output_fix.py src/gpuwrf/physics/noah_mp.py src/gpuwrf/io/land_state.py tests/test_m6_noah_mp_prescribed.py tests/savepoint/test_static_fields.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_X64=1 JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src pytest -q tests/test_m6_noah_mp_prescribed.py tests/savepoint/test_static_fields.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_tsk_znt_sourcing_fix.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_source_fidelity_closure.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/mynn_driver_source_output_fix.py`
- `python -m json.tool` on the proof JSONs.

## Results

- Focused tests: `4 passed, 1 skipped`.
- Primary proof JSON validated.
- Historical source-fidelity and MYNN proof JSONs validated after their
  next-route labels were updated.

## Residual Test Gap

The strict Step-1 field gate is still red and must be rerun after the next
thermodynamic-column input fix. No GPU validation or TOST should start from this
intermediate candidate.
