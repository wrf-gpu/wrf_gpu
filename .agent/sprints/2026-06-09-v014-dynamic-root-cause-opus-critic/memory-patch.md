# Memory Patch Proposal

## Scope

Project memory update for the v0.14 Opus dynamic root-cause critic.

## Evidence

- `.agent/reviews/2026-06-09-v014-dynamic-root-cause-opus-critic.md`
  concludes `MANAGER_FINAL_RK_TARGET_NOT_JUSTIFIED_INPUT_ALREADY_DIVERGED`.
- `proofs/v014/dynamic_root_cause_opus_critic.json` records the same verdict.
- Existing `proofs/v014/pre_rk_input_boundary.json` shows the JAX input to step
  6000 is already divergent before final RK.
- Existing `proofs/v014/grid_after_live_nest_base.json` proves the base-source
  fix did not close grid/V10 divergence.

## Proposed Destination

Create pending memory:

- `.agent/memory/pending/2026-06-09-v014-dynamic-root-cause-opus-critic.md`

Do not promote to stable memory until the strict same-input single-RK-step proof
confirms or refutes the critic's recommended boundary.

## Patch

Record that final-RK output instrumentation is not the next source-edit target
while the pre-RK input is already divergent. The next proof boundary is strict
same-input single-RK-step parity from WRF pre-RK input to WRF post-RK/pre-halo
output, with tendency control and stencil-valid patch scoring.

## Reviewer Status:

Pending. Accepted as a critic sprint; stable promotion waits for the next proof.
