# Pending Memory Patch: V0.14 Theta Prestep Input Boundary

Scope:

Project-memory update for v0.14 same-state JAX-vs-WRF grid-parity debugging
after theta-evolution localization.

Evidence:

- `proofs/v014/jax_theta_evolution_localization.json` reports
  `THETA_MISMATCH_PRESTEP_OR_INPUT`.
- The first reachable theta mismatch is already at the available WRF
  start-of-step/RK-reference theta surface (`T_OLD` / `grid%t_1`) versus the
  real JAX step-5999 prestep carry input: max_abs `6.218735851548047`, RMSE
  `4.638818160588427`.
- `MU_OLD` input-boundary context also differs, max_abs
  `267.01919069732367`.
- The proof-local RK mirror matches the existing pre-halo helper for theta with
  max_abs `0.0`, so this is not a proof-wrapper artifact.
- Current WRF source artifacts do not expose explicit step-6000 pre-RK
  `P/PB/MUB`, so source-changing dycore edits remain premature.

Proposed destination:

After the next input-boundary emitter/hook sprint, add a concise stable-memory
entry:

- When h10 `T` mismatch is already present at the earliest available
  start-of-step theta reference, do not start with final small-step finish,
  post-RK refresh, or history-source mapping fixes. First expose explicit
  step-6000 pre-RK WRF and JAX input-boundary `T/P/PB/MU/MUB` to distinguish
  bad checkpoint/prestep carry generation from a prior-step or packaging fault.

Reviewer Status:

Pending. Do not apply to stable memory until the explicit input-boundary sprint
confirms the narrower producer of the bad prestep state.
