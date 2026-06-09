# Memory Patch Proposal

## Scope

Project memory update for the v0.14 Step-1 RK1 source-boundary sprint.

## Evidence

- `proofs/v014/step1_rk1_source_boundary.json` records verdict
  `STEP1_RK1_SOURCE_LOCALIZED_FIRST_RK_STEP_PART1_PHYSICS_STATE_MUTATION_T_STATE`.
- The proof consumed WRF truth files emitted by disposable, env-gated scratch
  WRF instrumentation under
  `/mnt/data/wrf_gpu2/v014_step1_rk1_source_boundary/wrf_truth`.
- First localized source boundary is `after_first_rk_step_part1`, field
  `T_STATE`.
- WRF vs JAX operational carry max_abs is `5.490173101425171`, RMSE
  `1.9175184863907806`.
- WRF vs `_physics_step_forcing.state` max_abs is `5.490142455570492`, RMSE
  `1.9174736017582765`.
- RK1 `small_step_prep` continuity remains exact for `T_WORK` and `P_WORK`,
  both max_abs `0.0`.
- Manager validation reran the CPU proof, JSON validation, Python compilation,
  and confirmed `git diff -- src/gpuwrf` is empty.

## Proposed Destination

Create pending memory:

- `.agent/memory/pending/2026-06-09-v014-step1-rk1-source-boundary.md`

Also update:

- `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`
- `.agent/decisions/V0140-VALIDATION-PLAN.md`

## Patch

Record that v0.14 grid-parity debugging has localized the remaining Step-1
T-state mismatch to WRF `first_rk_step_part1` output. The next target is
internal `first_rk_step_part1` instrumentation/comparison against the JAX
physics adapter output. Do not continue acoustic, TOST, Switzerland, FP32, or
memory work until this earlier grid-parity boundary is explained or fixed.

## Reviewer Status:

Pending. Accepted as sprint-local memory only.
