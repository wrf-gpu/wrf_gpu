# Tester Report

## Decision:

Pass. The proof is reproducible under CPU-only JAX, validates as JSON, and
leaves production `src/gpuwrf/**` unchanged.

## Manager Re-Run Commands

- `python -m py_compile proofs/v014/step1_pre_part1_handoff.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_pre_part1_handoff.py`
- `python -m json.tool proofs/v014/step1_pre_part1_handoff.json >/tmp/step1_pre_part1_handoff.manager.validated.json`
- `git diff -- src/gpuwrf`

## Results

- Python compilation passed.
- CPU proof rerun reproduced verdict
  `STEP1_PRE_PART1_LOCALIZED_JAX_LOADER_T_STATE`.
- JSON validation passed.
- `git diff -- src/gpuwrf` was empty.
- JSON records `cpu_only=true`, `gpu_used=false`,
  `production_src_edits=false`, no TOST, no Switzerland, no FP32 source work,
  no memory source work, and no Hermes.

## Coverage

The proof parses full d02 WRF solve_em pre-part1 surfaces and compares:

- WRF internal `T_STATE` deltas from step increment to pre-call;
- continuity to the prior part1-entry proof;
- raw JAX live-nest state/carry and haloed step-entry views;
- full-theta versus perturbation-theta semantics.

## Residual Risk

The proof localizes the mismatch to JAX live-nest loader/carry construction but
does not yet split that construction internally or apply a fix.
