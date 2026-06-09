# Worker Report

## Summary:

The sprint is closed fail-closed. The GPT worker stalled before writing
artifacts, so the manager produced the required proof objects from the existing
validated WRF surfaces. No strict same-input JAX comparison was run.

Final verdict:
`FULL_DOMAIN_WRAPPER_BLOCKED_TRUTH_SURFACE_PATCH_ONLY_AND_CARRY_LEAVES`.

## Files Changed

- `proofs/v014/full_domain_source_truth.py`
- `proofs/v014/full_domain_source_truth.json`
- `proofs/v014/full_domain_source_truth.md`
- `proofs/v014/full_domain_source_truth_wrf_patch.diff`
- `proofs/v014/same_input_single_rk_parity_wrapped.py`
- `proofs/v014/same_input_single_rk_parity_wrapped.json`
- `proofs/v014/same_input_single_rk_parity_wrapped.md`
- `.agent/reviews/2026-06-09-v014-full-domain-source-wrapper.md`

No production `src/gpuwrf/**` files were changed.

## Commands Run

- `python -m py_compile proofs/v014/full_domain_source_truth.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/full_domain_source_truth.py`
- `python -m json.tool proofs/v014/full_domain_source_truth.json >/tmp/full_domain_source_truth.manager.validated.json`
- `python -m py_compile proofs/v014/same_input_single_rk_parity_wrapped.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/same_input_single_rk_parity_wrapped.py`
- `python -m json.tool proofs/v014/same_input_single_rk_parity_wrapped.json >/tmp/same_input_single_rk_parity_wrapped.manager.validated.json`
- `git diff -- src/gpuwrf`
- `git diff --check -- proofs/v014/full_domain_source_truth.py proofs/v014/full_domain_source_truth.json proofs/v014/full_domain_source_truth.md proofs/v014/full_domain_source_truth_wrf_patch.diff proofs/v014/same_input_single_rk_parity_wrapped.py proofs/v014/same_input_single_rk_parity_wrapped.json proofs/v014/same_input_single_rk_parity_wrapped.md .agent/reviews/2026-06-09-v014-full-domain-source-wrapper.md`

## Proof Objects Produced

- `proofs/v014/full_domain_source_truth.json`
- `proofs/v014/full_domain_source_truth.md`
- `proofs/v014/full_domain_source_truth_wrf_patch.diff`
- `proofs/v014/same_input_single_rk_parity_wrapped.json`
- `proofs/v014/same_input_single_rk_parity_wrapped.md`
- `.agent/reviews/2026-06-09-v014-full-domain-source-wrapper.md`

## Findings

Existing source/save and post-RK/pre-halo surfaces are patch-only. The accepted
source/save proof reports one conservative 8-cell-halo-valid mass cell. The full
wrapper carry/boundary leaves are not emitted at the same boundary. A strict
same-input JAX run would therefore be a weak comparison and was not executed.

## Unresolved Risks

Step-6000 same-input parity remains unexecuted. Per Opus management review, this
path should not continue as another one-blocker micro-sprint.

## Next Decision Needed

Open the consolidated early-step same-input discriminator sprint from
`.agent/decisions/V0140-EARLY-STEP-DISCRIMINATOR-PLAN.md`.
