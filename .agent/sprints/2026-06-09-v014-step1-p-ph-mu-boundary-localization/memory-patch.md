# Memory Patch: V0.14 Step-1 P/PH/MU Boundary Localization

Date: 2026-06-09

Reviewer Status: APPROVED_FOR_PENDING_MEMORY.

Scope:

- Pending memory only after close. Do not edit stable memory from this opening
  sprint.

Reason:

- The sprint localized the current post-theta/QV `P/PH/MU` residual to the
  first checked P-family state mismatch after WRF `first_rk_step_part1`.
- No source fix was applied because a narrower internal WRF split is still
  needed before editing production code.

Evidence:

- `proofs/v014/step1_p_ph_mu_boundary_localization.json`
- `proofs/v014/step1_p_ph_mu_boundary_localization.md`
- `.agent/reviews/2026-06-09-v014-step1-p-ph-mu-boundary-localization.md`

Proposed Destination:

- `.agent/memory/pending/2026-06-09-v014-step1-p-ph-mu-boundary-localization.md`

Approved pending memory:

- Current first material P-family state residual is WRF
  `after_first_rk_step_part1` vs JAX `_physics_step_forcing.carry.state`,
  `P_STATE` max_abs `69.96875`; `MU_STATE` and `W_STATE` are material there too.
- RK1 `small_step_prep`/`calc_p_rho(step=0)` work arrays are exact for checked
  work fields.
- No source fix was made.
- Next sprint should emit an internal WRF `first_rk_step_part1` split around
  `phy_prep`/`calc_p_rho_phi` state writes for `P/MU/W`, or split
  post-acoustic/pre-refresh pressure before source edits.
