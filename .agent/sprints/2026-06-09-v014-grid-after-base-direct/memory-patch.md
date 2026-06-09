# Memory Patch Proposal

## Scope

Project memory update for the v0.14 direct grid-after-base proof.

## Evidence

- `proofs/v014/grid_after_live_nest_base.json` verdict is
  `GRID_SYMPTOM_NOT_CLOSED`.
- The GPU h12 run exited green (`L2_D02_GREEN`) and the CPU-only wrfout
  comparator completed over d02 h1-h12.
- Dynamic residuals remain large: `V10` RMSE `2.55039100124724`, `PSFC` RMSE
  `517.1905702423264` Pa, `P` RMSE `230.30713670774634` Pa, `MU` RMSE
  `266.52491970646497`, and `PH` RMSE `292.3872984317863`.
- Static/base fields improved materially, but PB/MUB are not exact and the
  grid symptom did not collapse.
- `proofs/v014/same_state_momentum_mass.json` independently found the selected
  h10 `U` mismatch at `post_after_all_rk_steps_pre_halo`.

## Proposed Destination

Create pending memory:

- `.agent/memory/pending/2026-06-09-v014-grid-after-base-direct.md`

Do not promote to stable memory until the next Opus critic/debugger and fresh
same-state carry sprint confirm the next root-cause direction.

## Patch

Record that the live-nest base-source fix improved static/base payloads but did
not close dynamic grid parity. The v0.14 root-cause path should focus on final
RK pressure-gradient/mass-wind/theta-pressure coupling, using fresh same-state
current-code carries before more long validation runs.

## Reviewer Status:

Pending. Accepted as sprint-local memory, not stable memory.
