# Tester Report

Decision: PASS_PARTIAL_NO_SOURCE_PATCH.

## Commands

Manager reran the worker's required CPU-only gates:

- `python -m py_compile proofs/v014/step1_live_nest_theta_semantics.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_live_nest_theta_semantics.py`
- `python -m json.tool proofs/v014/step1_live_nest_theta_semantics.json >/tmp/step1_live_nest_theta_semantics.manager.validated.json`
- `git diff -- src/gpuwrf`

## Results

- The script reproduced verdict
  `STEP1_LIVE_NEST_THETA_ADJUST_TEMPQV_PARTIAL_NEXT_TSTATE_MILLIKELVIN_RESIDUAL`.
- JSON validation passed.
- The proof records `gpu_used=false`.
- `git diff -- src/gpuwrf` was empty, confirming there was no production source
  edit.

## Coverage

The test covers the proof-local WRF semantic candidates, not a production model
patch. It verifies that the strongest candidate reduces but does not eliminate
the `T_STATE` residual under the current accepted truth schema.

## Residual Risk

The proof intentionally does not authorize a source patch. It also cannot close
`QVAPOR`, because accepted same-boundary WRF pre-call `QVAPOR` truth is absent.
