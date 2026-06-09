# Memory Patch Proposal

## Scope

Project memory update for the v0.14 Step-1 pre-part1 handoff sprint.

## Evidence

- `proofs/v014/step1_pre_part1_handoff.json` records verdict
  `STEP1_PRE_PART1_LOCALIZED_JAX_LOADER_T_STATE`.
- WRF solve_em pre-call truth was emitted by scratch-only env-gated WRF
  instrumentation under
  `/mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/wrf_truth`.
- WRF `T_STATE` delta from `after_step_increment` to
  `before_first_rk_step_part1_call` is max_abs `0.0`.
- WRF solve_em pre-call vs prior part1-entry `T_STATE` continuity is max_abs
  `0.0`.
- WRF pre-call `T_STATE` vs raw JAX live-nest input state
  (`State.theta - 300 K`) has max_abs `5.490173101425171`, RMSE
  `1.9175184863907806`.
- Full-vs-perturbation theta was explicitly checked and concluded
  `WRF_T_STATE_IS_PERTURBATION_THETA`.
- Manager validation reran the CPU proof, JSON validation, Python compilation,
  and confirmed `git diff -- src/gpuwrf` is empty.

## Proposed Destination

Create pending memory:

- `.agent/memory/pending/2026-06-09-v014-step1-pre-part1-handoff.md`

Also update:

- `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`
- `.agent/decisions/V0140-VALIDATION-PLAN.md`

## Patch

Record that v0.14 Step-1 `T_STATE` divergence is localized to the JAX
live-nest Step-1 loader/carry boundary before `_physics_step_forcing`. WRF
solve_em pre-call mutation, `first_rk_step_part1`, and full-vs-perturbation
theta mapping are ruled out for this residual. The next target is JAX
loader/carry construction, not more WRF solve_em/physics/acoustic work.

## Reviewer Status:

Pending. Accepted as sprint-local memory only.
