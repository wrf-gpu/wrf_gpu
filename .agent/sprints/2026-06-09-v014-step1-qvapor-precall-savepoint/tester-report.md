# Tester Report

Decision: PASS_SAVEPOINT_READY.

## Commands

Manager ran:

- `python -m py_compile proofs/v014/step1_qvapor_precall_savepoint.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_qvapor_precall_savepoint.py`
- `python -m json.tool proofs/v014/step1_qvapor_precall_savepoint.json >/tmp/step1_qvapor_precall_savepoint.manager.validated.json`
- `git diff -- src/gpuwrf`

## Results

- The proof reproduced verdict
  `STEP1_QVAPOR_PRECALL_SAVEPOINT_READY`.
- JSON validation passed.
- The proof records `gpu_used=false`.
- `git diff -- src/gpuwrf` was empty.
- Filtered pre-call root contains 28 files:
  `/mnt/data/wrf_gpu2/v014_step1_qvapor_precall_savepoint/precall_truth_only`.

## Coverage

The validator compares the new QVAPOR-bearing WRF dump against the accepted
pre-call dump tile by tile and record by record. It validates that old mass and
W/PH fields are text-identical, and that QVAPOR is full-shape and finite.

## Residual Risk

The WRF source change is scratch-only and not committed as production source.
The next proof still needs to rerun theta semantics with this exact QVAPOR
boundary; this test does not itself close the 0.0054 K theta tail.
