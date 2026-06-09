# V0.14 Source/Save Boundary

Status: pending review.

The v0.14 source/save-boundary sprint found a valid WRF `d02` step-6000 boundary
after `first_rk_step_part1`, `first_rk_step_part2`, and `rk_tendency`, but
before `relax_bdy_dry`, `rk_addtend_dry`, `spec_bdy_dry`, `small_step_prep`,
and `advance_uv`.

Proofs:

- `proofs/v014/source_save_boundary_hook.json`
- `proofs/v014/same_input_single_rk_parity_sources.json`

Verdicts:

- `SOURCE_SAVE_BOUNDARY_HOOK_READY`
- `SOURCE_SAVE_BOUNDARY_READY_NO_JAX_WRAPPER_FULL_DOMAIN_PATCH_AND_SCALAR_OLD_LIMITER`

Key fact: WRF current-step dry source/save leaves are now emitted at a valid
pre-mutation boundary, and native dry state preservation versus the full pre-RK
savepoint is exact on overlap. Grid parity remains blocked because the repo
lacks a proof-only full-domain wrapper/truth surface and a consistent old-field
strategy for the selected comparison boundary.

Next exact target: full-domain/full-vertical WRF source/save plus post-RK truth
surface and the narrowest proof-only JAX wrapper that can execute one
same-input RK step with WRF-controlled inputs.
