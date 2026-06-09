# Memory Patch Proposal

## Scope

Project memory update for the v0.14 same-input single-RK parity sprint.

## Evidence

- `proofs/v014/same_input_single_rk_parity.json` records verdict
  `SAME_INPUT_TENDENCY_INPUT_BLOCKED_PRE_RK_FULL_NATIVE_STATE_RK_TENDF_AND_HISTORY_SOURCE_FIELDS`.
- `proofs/v014/same_input_single_rk_parity.md` explains that the available WRF
  pre-RK hook emits only `MASS_K1` T/P/PB/MU/MUB-related fields.
- `.agent/reviews/2026-06-09-v014-same-input-single-rk-parity.md` accepts the
  result as blocked instrumentation, not as a causal dynamics proof.
- `git diff -- src` is empty for the sprint.

## Proposed Destination

Create pending memory:

- `.agent/memory/pending/2026-06-09-v014-same-input-single-rk-parity.md`

Do not promote to stable memory until the full pre-RK hook and same-input parity
comparison complete.

## Patch

Record that the strict same-input single-RK proof cannot be executed from the
current WRF pre-RK savepoint because it lacks full native state, controlled base
tendencies, WRF dry-physics/source tendencies, and an `OperationalCarry` loader.
The next proof-enabling sprint is a full pre-RK native-state/tendency hook, not a
production dycore source edit.

## Reviewer Status:

Pending. Accepted as sprint-local memory only.
