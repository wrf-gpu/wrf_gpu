# Tester Report

Decision: PASS_MISSING_TRUTH_SPEC_READY.

## Commands

Manager ran the required CPU-only gates:

- `python -m py_compile proofs/v014/step1_qvapor_precall_truth_schema.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_qvapor_precall_truth_schema.py`
- `python -m json.tool proofs/v014/step1_qvapor_precall_truth_schema.json >/tmp/step1_qvapor_precall_truth_schema.manager.validated.json`
- `git diff -- src/gpuwrf`

## Results

- The script reproduced verdict
  `STEP1_QVAPOR_PRECALL_TRUTH_MISSING_SAVEPOINT_SPEC_READY`.
- JSON validation passed.
- The proof records `gpu_used=false`.
- `git diff -- src/gpuwrf` was empty.

## Coverage

The validator inventories the accepted pre-call truth schema and existing
QVAPOR-bearing Step-1 artifacts. It distinguishes same-boundary pre-call truth
from post-RK/pre-halo artifacts.

## Residual Risk

This sprint does not emit the missing WRF truth. It only proves that the truth
is missing and specifies the minimal savepoint needed next.
