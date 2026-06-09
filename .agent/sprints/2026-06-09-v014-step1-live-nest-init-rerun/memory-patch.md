# Memory Patch Proposal

## Scope

Project memory update for the v0.14 Step-1 live-nest initialization rerun.

## Evidence

- `proofs/v014/step1_live_nest_init_rerun.json` records verdict
  `STEP1_LIVE_NEST_INIT_BASE_RESIDUALS_CLOSED_NEXT_T`.
- The strict live-nest-init comparison executed against the accepted CPU-WRF d02
  Step-1 `post_after_all_rk_steps_pre_halo` truth npz.
- Base residuals collapsed to small thresholds:
  `MUB` max_abs `0.05002361937658861`, `PB` max_abs
  `0.05357326504599769`, `PHB` max_abs `0.10811684231157415`.
- The comparison still diverges, with first divergent field `T` and largest
  residual `P` max_abs `1561.2503728885986`.
- Manager validation reran the CPU proof, JSON validation, Python compilation,
  and confirmed `git diff -- src/gpuwrf` is empty.

## Proposed Destination

Create pending memory:

- `.agent/memory/pending/2026-06-09-v014-step1-live-nest-init-rerun.md`

Also update:

- `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`
- `.agent/decisions/V0140-VALIDATION-PLAN.md`

## Patch

Record that the v0.14 grid-parity debug ladder has closed the Step-1
live-nest/base initialization residual as the dominant cause. The active
remaining Step-1 problem is dynamic/operator localization: first divergent
schema field `T`, dominant residual `P`, with `PH/MU/W` also materially off.

TOST, Switzerland, FP32, and memory follow-ups stay behind this gate.

## Reviewer Status:

Pending. Accepted as sprint-local memory only.
