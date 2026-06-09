# Tester Report: V0.14 Step-1 Adjust-TempQV Intermediate Truth

Date: 2026-06-09

Decision: PASS for the sprint gate. The proof is CPU-only, uses the successful
manager WRF run rather than the earlier sandbox PMIx failure, emits the required
numeric comparison, and leaves production `src/gpuwrf/**` untouched.

## Validation Commands

```bash
python -m py_compile proofs/v014/step1_adjust_tempqv_intermediate.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/step1_adjust_tempqv_intermediate.py
python -m json.tool proofs/v014/step1_adjust_tempqv_intermediate.json \
  >/tmp/step1_adjust_tempqv_intermediate.manager.validated.json
git diff -- src/gpuwrf
```

All commands completed successfully. `git diff -- src/gpuwrf` was empty.

## Key Evidence

- Proof verdict:
  `STEP1_ADJUST_TEMPQV_INTERMEDIATE_PRESSURE_INPUT_MISMATCH`
- WRF run return code: `0`
- WRF hook status: `READY`
- GPU used: `False`
- Max pressure/input delta among compared fields:
  `17.67503987130476 Pa`
- `t_2_post` absolute delta:
  `0.00541785382188209 K`

## Test Risk

Only one target cell was emitted. That is sufficient for this sprint's
classification of the already-known worst residual path, but not sufficient as
a broad production fix proof.
