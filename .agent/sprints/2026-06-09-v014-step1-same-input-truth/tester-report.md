# Tester Report

## Decision:

Pass. The proof is reproducible under CPU-only JAX, emits valid JSON, and leaves
production `src/gpuwrf/**` unchanged.

## Manager Re-Run Commands

- `python -m py_compile proofs/v014/step1_same_input_truth.py`
- `python -m json.tool proofs/v014/step1_same_input_truth.json >/tmp/step1_same_input_truth.manager.prevalidated.json`
- `git diff -- src/gpuwrf`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_same_input_truth.py`
- `python -m json.tool proofs/v014/step1_same_input_truth.json >/tmp/step1_same_input_truth.manager.validated.json`

## Results

- Python compilation passed.
- JSON validation passed.
- Re-run reproduced verdict:
  `STEP1_SAME_INPUT_COMPARISON_EXECUTED_FIRST_DIVERGENT_T`.
- CPU-only environment was recorded in JSON: `JAX_PLATFORMS=cpu`,
  `CUDA_VISIBLE_DEVICES=""`, `jax_default_backend="cpu"`, and
  `gpu_device_count=0`.
- `git diff -- src/gpuwrf` was empty.
- No WRF/MPI process remained after the worker run.

## Coverage

The strict comparison covers the full d02 domain for all 16 schema fields:
`T/P/PB/PH/PHB/MU/MUB/U/V/W/QVAPOR/QCLOUD/QRAIN/QICE/QSNOW/QGRAUP`.

## Residual Risk

This is a first-divergence proof, not a fix. The next validation gate must rerun
the same proof after the base-state/live-nest initialization issue is fixed or
falsified.
