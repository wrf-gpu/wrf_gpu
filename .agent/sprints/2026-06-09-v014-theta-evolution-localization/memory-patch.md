# Memory Patch

Scope:

Project-memory update for v0.14 same-state JAX-vs-WRF grid-parity debugging
after theta-evolution localization.

Evidence:

- `proofs/v014/jax_theta_evolution_localization.json` reports
  `THETA_MISMATCH_PRESTEP_OR_INPUT`.
- First reachable theta mismatch is against WRF `T_OLD` /
  `grid%t_1`, with max_abs `6.218735851548047` and RMSE
  `4.638818160588427`.
- `MU_OLD` input-boundary context also differs, max_abs
  `267.01919069732367`.
- The proof-local RK mirror matches the existing pre-halo helper for theta with
  max_abs `0.0`.
- The current WRF artifacts do not expose explicit step-6000 pre-RK `P/PB/MUB`,
  so a production source fix remains premature.

Proposed destination:

Create `.agent/memory/pending/2026-06-09-v014-theta-prestep-input-boundary.md`.
After the next input-boundary emitter/hook sprint, condense the durable lesson
into `.agent/memory/stable/recurring-gotchas.md`.

Reviewer Status:

Pending. Do not apply to stable memory until the explicit step-6000 pre-RK
input-boundary sprint confirms whether the bad state originates in JAX
checkpoint/prestep carry generation, prior-step update, or boundary/tendency
packaging.
