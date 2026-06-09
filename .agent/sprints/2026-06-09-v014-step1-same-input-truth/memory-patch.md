# Memory Patch Proposal

## Scope

Project memory update for the v0.14 step-1 same-input truth sprint.

## Evidence

- `proofs/v014/step1_same_input_truth.json` records verdict
  `STEP1_SAME_INPUT_COMPARISON_EXECUTED_FIRST_DIVERGENT_T`.
- The accepted comparison is WRF d02 step-1 post-RK/pre-halo truth against JAX
  one-step `_rk_scan_step_with_pre_halo_capture(...).pre_halo_state`, not the
  JAX initial state.
- The full-domain truth npz exists:
  `/mnt/data/wrf_gpu2/v014_same_input_contract_builder/wrf_truth/same_input_post_after_all_rk_steps_pre_halo_d02_step_1.npz`.
- Top residuals are dominated by `MUB/PB/PHB/P`, with `MUB` max_abs
  `2635.640625` and `PB` max_abs `2627.3828125`.
- Manager validation reran the CPU proof, JSON validation, Python compilation,
  and confirmed `git diff -- src/gpuwrf` is empty.

## Proposed Destination

Create pending memory:

- `.agent/memory/pending/2026-06-09-v014-step1-same-input-truth.md`

Also update:

- `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`
- `.agent/decisions/V0140-VALIDATION-PLAN.md`

## Patch

Record that v0.14 grid-parity debugging now has a real first-divergence
full-domain proof at d02 step 1. The next sprint should target native live-nest
child base-state initialization or a decisive init-override falsifier, then rerun
`proofs/v014/step1_same_input_truth.py`.

## Reviewer Status:

Pending. Accepted as sprint-local memory only.
