# Memory Patch

Scope:

Project-memory update for v0.14 same-state grid-parity debugging after explicit
pre-RK input-boundary evidence.

Evidence:

- `proofs/v014/pre_rk_input_boundary.json` reports
  `PRE_RK_INPUT_JAX_PRESTEP_MISMATCH_CONFIRMED`.
- CPU-WRF d02 h10 step-6000 pre-RK truth was emitted before `cpl_store_input`
  and before current-step physics/RK.
- The produced JAX h10 step-5999 carry differs from that WRF boundary for all
  target fields: `T/P/PB/MU/MUB`.
- The first mismatch is `T` max_abs `6.218735851548047`, RMSE
  `4.638818160588427`.
- `P`, `PB`, `MU`, and `MUB` also differ with large max_abs values, so this is
  not a theta-only source mapping issue.

Proposed destination:

Create `.agent/memory/pending/2026-06-09-v014-pre-rk-input-boundary.md`.
After the next checkpoint/prestep producer trace, condense the durable lesson
into stable memory only if it generalizes beyond this h10 debug chain.

Reviewer Status:

Pending. Do not apply to stable memory until the next sprint identifies the
first wrong JAX write or confirms this as a recurring checkpoint/handoff rule.
