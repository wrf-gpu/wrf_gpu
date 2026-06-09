# Tester Report

## Decision:

Pass. The proof is reproducible under CPU-only JAX, validates as JSON, and
leaves production `src/gpuwrf/**` unchanged.

## Manager Re-Run Commands

- `python -m py_compile proofs/v014/step1_part1_physics_state_mutation.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_part1_physics_state_mutation.py`
- `python -m json.tool proofs/v014/step1_part1_physics_state_mutation.json >/tmp/step1_part1_physics_state_mutation.manager.validated.json`
- `git diff -- src/gpuwrf`

## Results

- Python compilation passed.
- CPU proof rerun reproduced verdict
  `STEP1_PART1_INPUT_ALREADY_DIVERGED_T_STATE`.
- JSON validation passed.
- `git diff -- src/gpuwrf` was empty.
- JSON records `cpu_only=true`, `gpu_used=false`,
  `production_src_edits=false`, no TOST, no Switzerland, no FP32 source work,
  no memory source work, and no Hermes.

## Coverage

The proof parses full d02 WRF internal `first_rk_step_part1` surfaces emitted by
scratch WRF instrumentation and compares `T_STATE` at call entry, internal
boundaries, and exit against the JAX live-nest step-entry and
`_physics_step_forcing` surfaces.

## Residual Risk

The proof localizes the mismatch upstream of `first_rk_step_part1` but does not
identify the upstream handoff source or apply a production fix.
