# Memory Patch Proposal

## Scope

Project memory update for the v0.14 early-step same-input discriminator sprint.

## Evidence

- `proofs/v014/early_step_discriminator.json` records verdict
  `EARLY_STEP_DISCRIMINATOR_BLOCKED_CPU_REALCASE_LOADER_GPU_ONLY_NO_CANDIDATE_WRF_PREHALO_TRUTH_NO_SAME_INPUT_CARRY_CONTRACT`.
- Candidate steps `1`, `60`, `600`, `3000`, and `5999` are all blocked by the
  same missing same-input contracts.
- The production real-case loader reaches `State.zeros`, which requires a
  visible GPU under CPU-only proof rules.
- No candidate-step WRF post-RK/pre-halo full-field surface exists.
- `git diff -- src/gpuwrf` is empty.

## Proposed Destination

Create pending memory:

- `.agent/memory/pending/2026-06-09-v014-early-step-discriminator.md`

Also update the local manager skill to require a top-level "right debugging
tool?" check before long runtime-chasing debug ladders.

## Patch

Record that v0.14 grid-parity debugging is now blocked by missing same-input
comparison infrastructure, not by a proven dynamics source line. The next debug
sprint should build the CPU-compatible proof loader and WRF candidate-step
truth surface before further bisection or source edits.

## Reviewer Status:

Pending. Accepted as sprint-local memory only.
