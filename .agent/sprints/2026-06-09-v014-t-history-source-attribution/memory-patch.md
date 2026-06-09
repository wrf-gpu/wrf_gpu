# Memory Patch

Scope:

Project-memory update for v0.14 same-state JAX-vs-WRF debugging after the T
history/source-attribution sprint.

Evidence:

- `proofs/v014/jax_t_history_source_attribution.json` reports
  `T_EVOLUTION_MISMATCH_CONFIRMED`.
- The best JAX candidate for WRF history `T_HIST_SRC` is
  `captured_pre_halo_state.theta_minus_300`, still max_abs
  `3.3545763228707983`.
- The best JAX candidate for WRF `T_THM` is
  `captured_final_carry.t_2ave_minus_300`, still max_abs
  `3.677881697025043`.
- The proof confirms the h10 checkpoint identity matches both the producer and
  canonical h10 comparison records, and separates `T_HIST_SRC` from `T_THM`.

Proposed destination:

Create `.agent/memory/pending/2026-06-09-v014-t-evolution-mismatch.md`.
After independent review and after the theta-evolution localization sprint,
condense the durable lesson into `.agent/memory/stable/recurring-gotchas.md`.

Reviewer Status:

Pending. Do not apply to stable memory until the next localization sprint
identifies the failing theta stage/operator or proves that a broader mass/theta
state boundary is responsible.
