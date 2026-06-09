# Memory Patch Proposal

## Scope

Project memory update for the v0.14 same-state momentum/mass localization
sprint.

## Evidence

- `proofs/v014/same_state_momentum_mass.json` verdict is
  `JAX_MISMATCH_U_post_after_all_rk_steps_pre_halo`.
- The first failing field is `U`, max_abs `6.292358893898424`, RMSE
  `2.032497018496295`, against WRF's `post_after_all_rk_steps_pre_halo` text
  surface.
- The proof is CPU-only, JSON-valid, and has no production `src/` diff.
- The h10 carry predates `proofs/v014/live_nest_base_source_fix.json`, so base
  field residuals need a regenerated carry before attribution.

## Proposed Destination

Create pending memory:

- `.agent/memory/pending/2026-06-09-v014-same-state-momentum-mass.md`

Do not promote to stable memory until the next fresh-carry localization confirms
whether the post-RK/pre-halo `U` mismatch survives current-code regeneration.

## Patch

Record that the v0.14 grid-divergence search has a named same-state dynamic
surface failure: selected h10 `U` mismatches WRF before RK halo exchange and
before output/station validation. The next debug target is one layer earlier
inside final RK U/V tendency/acoustic update, mass coupling, and theta-pressure
source assembly.

## Reviewer Status:

Pending. The artifact is accepted as a localization proof, but stable memory
promotion should wait for the fresh h10 carry after the live-nest base-source
partial fix.
