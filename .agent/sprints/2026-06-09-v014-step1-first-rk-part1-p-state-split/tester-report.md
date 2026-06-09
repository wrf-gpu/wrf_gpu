# Tester Report: V0.14 Step-1 First-RK Part1 P-State Split

Date: 2026-06-09

Decision: accept the proof validation as passing.

## Commands

- `python -m py_compile proofs/v014/step1_first_rk_part1_p_state_split.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_first_rk_part1_p_state_split.py`
- `python -m json.tool proofs/v014/step1_first_rk_part1_p_state_split.json >/tmp/step1_first_rk_part1_p_state_split.manager.validated.json`
- `git diff -- src/gpuwrf`

## Result

- Python compile passed.
- CPU proof rerun passed and reproduced verdict
  `STEP1_FIRST_RK_PART1_P_STATE_LOCALIZED_PRE_PART1_RAW_CHILD_STATE`.
- JSON validation passed.
- `git diff -- src/gpuwrf` had no output.

## Scope

No GPU, TOST, Switzerland, FP32 source work, memory source work, Hermes, or
production source edit was used.
