# Tester Report: V0.14 Step-1 JAX Start-Domain Input Split

Decision: `PASS_LOCALIZATION_NO_SOURCE_PATCH`.

Validated artifacts:

- `proofs/v014/step1_jax_start_domain_input_split.py`
- `proofs/v014/step1_jax_start_domain_input_split.json`
- `proofs/v014/step1_jax_start_domain_input_split.md`
- `.agent/reviews/2026-06-09-v014-step1-jax-start-domain-input-split.md`

Manager validation commands:

- `python -m py_compile proofs/v014/step1_jax_start_domain_input_split.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_jax_start_domain_input_split.py`
- `python -m json.tool proofs/v014/step1_jax_start_domain_input_split.json >/tmp/step1_jax_start_domain_input_split.manager.validated.json`
- `git diff -- src/gpuwrf`

Result:

- Python compile passed.
- CPU-only proof rerun passed and rewrote JSON/MD with verdict
  `STEP1_JAX_START_DOMAIN_INPUT_SPLIT_LOCALIZED_BASE_STATE_RECONSTRUCTION_FP32_ALT_SOURCE_ORDER_GAP`.
- JSON validation passed.
- `git diff -- src/gpuwrf` was empty, confirming no production source edit.

Key checked metrics:

- Current pressure residual: max_abs `3.9458582235092763`, RMSE
  `0.3832298992869327`.
- WRF ALT substitution residual: max_abs `0.07605321895971429`, RMSE
  `0.006830944106223064`.
- FP32 ALT diagnosis with WRF `PHB+MUB`: pressure max_abs `0.0859375`, RMSE
  `0.009877167668418278`.
- Best proof-local WRF-order fp32/cp=1004.5 base candidate remains above gate:
  `P_STATE` max_abs `2.828125`, `MU_STATE` max_abs `0.011962890625`.

Residual risk:

This is a localization result, not a source fix. The proof explicitly rejects a
production patch until the WRF base-state reconstruction/source-order contract
is closed.
