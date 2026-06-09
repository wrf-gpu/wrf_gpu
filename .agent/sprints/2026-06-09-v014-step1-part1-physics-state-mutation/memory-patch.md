# Memory Patch Proposal

## Scope

Project memory update for the v0.14 Step-1 part1 physics-state mutation sprint.

## Evidence

- `proofs/v014/step1_part1_physics_state_mutation.json` records verdict
  `STEP1_PART1_INPUT_ALREADY_DIVERGED_T_STATE`.
- The proof consumed WRF truth files emitted by disposable, env-gated scratch
  WRF instrumentation under
  `/mnt/data/wrf_gpu2/v014_step1_part1_physics_state_mutation/wrf_truth`.
- `part1_entry_before_init_zero_tendency` `T_STATE` vs JAX live-nest
  step-entry state has max_abs `5.490173101425171`, RMSE
  `1.9175184863907806`.
- Largest WRF internal `T_STATE` delta from part1 entry is max_abs `0.0`.
- Manager validation reran the CPU proof, JSON validation, Python compilation,
  and confirmed `git diff -- src/gpuwrf` is empty.

## Proposed Destination

Create pending memory:

- `.agent/memory/pending/2026-06-09-v014-step1-part1-physics-state-mutation.md`

Also update:

- `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`
- `.agent/decisions/V0140-VALIDATION-PLAN.md`

## Patch

Record that v0.14 Step-1 `T_STATE` divergence is already present at WRF
`first_rk_step_part1` entry and is not produced by part1's physics driver
sequence. The next target is the upstream live-nest/WRF call-site handoff before
`first_rk_step_part1`, not radiation, surface, PBL, cumulus, acoustic, TOST,
Switzerland, FP32, or memory work.

## Reviewer Status:

Pending. Accepted as sprint-local memory only.
