# Tester Report: V0.14 Step-1 Current-MUB/Base-Input Split

Date: 2026-06-09

Decision: PASS for the sprint gate. The proof is CPU-only, uses accepted WRF
truth plus a proof-side live-nest blend recompute, validates JSON/pycompile,
and leaves `src/gpuwrf/**` unchanged.

## Validation Commands

```bash
python -m py_compile proofs/v014/step1_current_mub_base_input_split.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/step1_current_mub_base_input_split.py
python -m json.tool proofs/v014/step1_current_mub_base_input_split.json \
  >/tmp/step1_current_mub_base_input_split.manager.validated.json
git diff -- src/gpuwrf
```

All manager validation commands completed successfully. `git diff -- src/gpuwrf`
was empty.

## Key Evidence

- Verdict:
  `STEP1_CURRENT_MUB_BASE_SPLIT_WRF_BLEND_UNIMPLEMENTED_OR_MISMATCHED`
- GPU used: `False`
- Required ancestor `9a7016d9`: present
- Comparison status: `COMPARISON_EXECUTED`
- WRF adjust current MUB minus JAX theta-proof final MUB:
  `17.67503987130476 Pa`
- WRF adjust current MUB minus JAX direct WRF-blend MUB:
  `-0.00045210951066110283 Pa`
- WRF pre-part1 final MUB minus JAX theta-proof final MUB:
  `-0.004647628695238382 Pa`

## Test Risk

The fresh WRF hook proposed by the worker was not run because the worker's
sandbox could not write `/mnt/data`. The accepted scalar truth is still enough
to classify the boundary mismatch, but the source-changing sprint must include
a field-level/full-domain proof before patch acceptance.
