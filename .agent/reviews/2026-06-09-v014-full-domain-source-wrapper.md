# V0.14 Full-Domain Source Wrapper Handoff

## objective

Close the full-domain wrapper sprint without a weak comparison after the GPT
worker stalled before writing artifacts. Inventory the existing source/save and
post-RK surfaces and emit the precise blocker.

## files changed

- `proofs/v014/full_domain_source_truth.py`
- `proofs/v014/full_domain_source_truth.json`
- `proofs/v014/full_domain_source_truth.md`
- `proofs/v014/full_domain_source_truth_wrf_patch.diff`
- `proofs/v014/same_input_single_rk_parity_wrapped.py`
- `proofs/v014/same_input_single_rk_parity_wrapped.json`
- `proofs/v014/same_input_single_rk_parity_wrapped.md`
- `.agent/reviews/2026-06-09-v014-full-domain-source-wrapper.md`

No production `src/gpuwrf/**` edits.

## commands run

- `python -m py_compile proofs/v014/full_domain_source_truth.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/full_domain_source_truth.py`
- `python -m json.tool proofs/v014/full_domain_source_truth.json >/tmp/full_domain_source_truth.validated.json`
- `python -m py_compile proofs/v014/same_input_single_rk_parity_wrapped.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/same_input_single_rk_parity_wrapped.py`
- `python -m json.tool proofs/v014/same_input_single_rk_parity_wrapped.json >/tmp/same_input_single_rk_parity_wrapped.validated.json`
- `git diff -- src/gpuwrf`

## proof objects produced

- `proofs/v014/full_domain_source_truth.json`
- `proofs/v014/full_domain_source_truth.md`
- `proofs/v014/full_domain_source_truth_wrf_patch.diff`
- `proofs/v014/same_input_single_rk_parity_wrapped.json`
- `proofs/v014/same_input_single_rk_parity_wrapped.md`

Truth-surface verdict:
`FULL_DOMAIN_TRUTH_SURFACE_BLOCKED_PATCH_ONLY_EXISTING_SURFACES`.

Wrapper verdict:
`FULL_DOMAIN_WRAPPER_BLOCKED_TRUTH_SURFACE_PATCH_ONLY_AND_CARRY_LEAVES`.

## unresolved risks

- No strict same-input JAX comparison executed at step 6000.
- Existing WRF source/save and post-RK/pre-halo surfaces are patch-only.
- Same-boundary full wrapper carry/boundary leaves are not emitted.
- The h10/step-6000 path is not the fastest next discriminator after Opus review.

## next decision needed

Open the consolidated early-step same-input discriminator sprint from
`.agent/decisions/V0140-EARLY-STEP-DISCRIMINATOR-PLAN.md`.
