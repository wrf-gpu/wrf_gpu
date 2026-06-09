# Manager Closeout: V0.14 Step-1 JAX Start-Domain Input Split

Date: 2026-06-09 20:32 WEST

## Outcome

Closed with verdict
`STEP1_JAX_START_DOMAIN_INPUT_SPLIT_LOCALIZED_BASE_STATE_RECONSTRUCTION_FP32_ALT_SOURCE_ORDER_GAP`.

Merge Decision: commit and push the proof artifacts. Do not merge a production
source change from this sprint.

The dominant residual is now localized to WRF base-state reconstruction feeding
fp32 `AL/ALT` diagnosis. The path is much narrower than the earlier broad
`T/P/MU/W` divergence, but a production patch is not yet justified because the
proof-local WRF-order base candidate still misses the `P/MU` gates.

## Proof Objects

- `proofs/v014/step1_jax_start_domain_input_split.py`
- `proofs/v014/step1_jax_start_domain_input_split.json`
- `proofs/v014/step1_jax_start_domain_input_split.md`
- `.agent/reviews/2026-06-09-v014-step1-jax-start-domain-input-split.md`

## Key Metrics

- Current pressure formula versus WRF P:
  max_abs `3.9458582235092763`, RMSE `0.3832298992869327`.
- Replacing diagnosed ALT with WRF ALT:
  max_abs `0.07605321895971429`, RMSE `0.006830944106223064`.
- FP32 ALT diagnosis with WRF `PHB+MUB`:
  pressure max_abs `0.0859375`, RMSE `0.009877167668418278`.
- WRF fields with fp64 ALT diagnosis:
  pressure max_abs `2.961779549412313`, showing dtype/order matters.
- Best local WRF-order fp32/cp=1004.5 base candidate:
  `P_STATE` max_abs `2.828125`, `MU_STATE` max_abs `0.011962890625`.

## Validation

Manager reran:

- `python -m py_compile proofs/v014/step1_jax_start_domain_input_split.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_jax_start_domain_input_split.py`
- `python -m json.tool proofs/v014/step1_jax_start_domain_input_split.json >/tmp/step1_jax_start_domain_input_split.manager.validated.json`
- `git diff -- src/gpuwrf` with no output

## Scope

No production source, GPU validation, TOST, Switzerland, FP32 source work,
memory source work, or Hermes was used.

## Next Sprint

Open a narrow WRF base-state boundary sprint. Emit or reproduce exact
`start_domain_em` base-state values before the hypsometric `AL/ALT` pass:
`p_surf`, post-assignment `MUB`, `PB/T_INIT/ALB`, `PHB`, active hybrid
coefficients, flags, and scalar constants. Once that base reconstruction
matches WRF `PHB+MUB`, apply the already-proven `P/MU/W` perturbation init path
under a source-changing sprint.
