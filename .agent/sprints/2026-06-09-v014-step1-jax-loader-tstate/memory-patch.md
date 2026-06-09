# Memory Patch Proposal

## Scope

Project memory update for closing the v0.14 Step-1 JAX loader `T_STATE` sprint.

## Evidence

- `proofs/v014/step1_jax_loader_tstate.json` records verdict
  `STEP1_JAX_LOADER_TSTATE_LOCALIZED_LIVE_NEST_STATE_BASE_MISMATCH`.
- `T_STATE` max_abs versus WRF pre-call is `5.490173101425171` for raw, live,
  boundary-packaged, carry, and haloed step-entry states.
- `T_STATE` transition max_abs is `0.0` for raw->live, live->boundary,
  boundary->carry, and carry->halo.
- `PB` improves from raw max_abs `2627.3828125` to live max_abs
  `0.05357326504599769`.
- Haloed step-entry interior max_abs is `5.490173101425171`, so the residual is
  not boundary-only.

## Proposed Destination

Create pending memory:

- `.agent/memory/pending/2026-06-09-v014-step1-jax-loader-tstate.md`

Also update:

- `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`
- `.agent/decisions/V0140-VALIDATION-PLAN.md`

## Patch

Record that v0.14 grid-parity debugging has localized the Step-1 `T_STATE`
residual to WRF live-nest initialization semantics. JAX live-nest base init
updates `PB/PHB/MUB` but leaves `State.theta` as raw `wrfinput_d02`; WRF pre-call
truth reflects a matching `t_2`/theta path after live-nest base setup. Boundary
package, initial carry, and halo are ruled out for this residual.

## Reviewer Status:

Accepted as sprint-local memory. Next target is WRF `med_nest_initial` /
`start_domain_em` `t_2` semantics.
