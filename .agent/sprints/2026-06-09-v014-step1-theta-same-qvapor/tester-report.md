# Tester Report

Decision: PASS_INTERIOR_RESIDUAL_PROVEN.

## Commands

Manager ran:

- `python -m py_compile proofs/v014/step1_theta_same_qvapor.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_theta_same_qvapor.py`
- `python -m json.tool proofs/v014/step1_theta_same_qvapor.json >/tmp/step1_theta_same_qvapor.manager.validated.json`
- `git diff -- src/gpuwrf`

## Results

- The proof reproduced verdict
  `STEP1_THETA_SAME_QVAPOR_INTERIOR_RESIDUAL_NEEDS_WRF_INTERMEDIATE`.
- JSON validation passed.
- The proof records `gpu_used=false`.
- `git diff -- src/gpuwrf` was empty.

## Coverage

The test covers same-boundary QVAPOR root parsing, all-cell final candidate
metrics, boundary versus interior decomposition, worst-cell detail, and QVAPOR
truth comparison. It does not run WRF, GPU, TOST, Switzerland, FP32 source work,
or memory source work.

## Residual Risk

The proof establishes that QVAPOR truth is no longer the blocker, but it does
not explain the remaining `0.00541785382188209 K` interior `T_STATE` residual.
The next test must emit WRF internals for that residual path.
