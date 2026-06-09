# Memory Patch Proposal

## Scope

Project memory update for opening the v0.14 Step-1 JAX loader `T_STATE` sprint.

## Evidence

- `proofs/v014/step1_pre_part1_handoff.json` records verdict
  `STEP1_PRE_PART1_LOCALIZED_JAX_LOADER_T_STATE`.
- WRF `T_STATE` is unchanged from `after_step_increment` to
  `before_first_rk_step_part1_call`, max_abs `0.0`.
- Full-vs-perturbation theta was checked and concluded
  `WRF_T_STATE_IS_PERTURBATION_THETA`.
- WRF pre-call `T_STATE` vs raw JAX live-nest state (`State.theta - 300 K`) has
  max_abs `5.490173101425171`, RMSE `1.9175184863907806`.

## Proposed Destination

Create pending memory:

- `.agent/memory/pending/2026-06-09-v014-step1-jax-loader-tstate.md`

Also update:

- `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`
- `.agent/decisions/V0140-VALIDATION-PLAN.md`

## Patch

Record that the active v0.14 grid-parity target is the JAX live-nest Step-1
loader/carry construction for `T_STATE`. The accepted proof ruled out WRF
pre-call mutation, `first_rk_step_part1`, and theta semantic offset. The next
stage split is `raw_child_state -> live_child_state -> boundary_package ->
initial_carry -> haloed_step_entry`.

## Reviewer Status

Pending. Opening sprint only.
