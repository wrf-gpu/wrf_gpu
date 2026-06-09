# V0.14 Same-Input Single-RK Wrapped Gate

Verdict: `FULL_DOMAIN_WRAPPER_BLOCKED_TRUTH_SURFACE_PATCH_ONLY_AND_CARRY_LEAVES`.

No JAX step was executed. A weak comparison was avoided because the available WRF surfaces do not satisfy the full-domain same-input wrapper contract.

## Blockers

- existing source/save surface is patch-only, not full-domain/full-vertical enough for the wrapper contract
- existing post-RK/pre-halo truth is patch-only and not a full-domain/full-vertical State truth surface
- accepted source/save proof reports only one conservative 8-cell-halo-valid mass cell
- no same-boundary promoted carry/boundary surface was emitted for the full wrapper contract

Next: use the staged early-step discriminator and bisect from shared `wrfinput`.
