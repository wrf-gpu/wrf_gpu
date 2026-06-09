# Memory Patch Proposal

## Scope

Project memory update for opening the v0.14 Step-1 live-nest theta semantics
sprint.

## Evidence

- `proofs/v014/step1_jax_loader_tstate.json` records verdict
  `STEP1_JAX_LOADER_TSTATE_LOCALIZED_LIVE_NEST_STATE_BASE_MISMATCH`.
- WRF source shows live-nest input-file initialization blends `ht/mub/phb`, then
  calls `adjust_tempqv(..., nest%t_2, nest%p, QVAPOR, use_theta_m, ...)`.
- Current JAX live-nest base init updates `PB/PHB/MUB` but leaves
  `State.theta` from raw `wrfinput_d02`.

## Proposed Destination

Create pending memory:

- `.agent/memory/pending/2026-06-09-v014-step1-live-nest-theta-semantics.md`

Also update:

- `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`
- `.agent/decisions/V0140-VALIDATION-PLAN.md`

## Patch

Record that the next active v0.14 grid-parity target is WRF
`adjust_tempqv` semantics after live-nest terrain/base blending. A production
fix is allowed only if a proof-local candidate closes WRF pre-call `T_STATE`
against the accepted truth.

## Reviewer Status:

Pending. Opening sprint only.
