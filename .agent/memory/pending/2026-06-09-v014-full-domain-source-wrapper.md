# V0.14 Full-Domain Source Wrapper

Status: pending review.

The v0.14 full-domain source-wrapper sprint closed fail-closed. Existing WRF
source/save and post-RK/pre-halo surfaces are patch-only and lack the
same-boundary full wrapper carry/boundary leaves needed for a strict step-6000
same-input JAX comparison.

Proofs:

- `proofs/v014/full_domain_source_truth.json`
- `proofs/v014/same_input_single_rk_parity_wrapped.json`

Verdicts:

- `FULL_DOMAIN_TRUTH_SURFACE_BLOCKED_PATCH_ONLY_EXISTING_SURFACES`
- `FULL_DOMAIN_WRAPPER_BLOCKED_TRUTH_SURFACE_PATCH_ONLY_AND_CARRY_LEAVES`

Next exact target: stop extending the h10/step-6000 wrapper ladder and run the
early-step same-input discriminator from shared `wrfinput`.
