# Tester Report

## Decision:

Pass. The manager rerun reproduced the worker verdict under CPU-only JAX, the
JSON proof validates, and production model source stayed unchanged.

## Manager Re-Run Commands

- `python -m py_compile proofs/v014/step1_live_nest_init_rerun.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_live_nest_init_rerun.py`
- `python -m json.tool proofs/v014/step1_live_nest_init_rerun.json >/tmp/step1_live_nest_init_rerun.manager.validated.json`
- `git diff -- src/gpuwrf`

## Results

- Python compilation passed.
- CPU proof rerun reproduced verdict
  `STEP1_LIVE_NEST_INIT_BASE_RESIDUALS_CLOSED_NEXT_T`.
- JSON validation passed.
- `git diff -- src/gpuwrf` was empty.
- The proof records `JAX_PLATFORMS=cpu`, `CUDA_VISIBLE_DEVICES=""`,
  `jax_default_backend="cpu"`, `gpu_device_count=0`, `gpu_used=false`, and
  `production_src_edits=false`.

## Coverage

The strict live-nest-init comparison executed for the frozen 16-field schema:
`T/P/PB/PH/PHB/MU/MUB/U/V/W/QVAPOR/QCLOUD/QRAIN/QICE/QSNOW/QGRAUP`.

## Residual Risk

The gate proves the base-state/live-nest initialization mismatch is no longer
the dominant Step-1 residual under this proof path. It does not yet prove the
next dynamic operator root cause.
