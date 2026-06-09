# Memory Patch Proposal: V0.14 Step-1 First-RK Part1 P-State Split

Date: 2026-06-09 18:57 WEST

Reviewer Status: Pending. Opening sprint only.

Scope:

- `.agent/memory/pending/v014-grid-parity.md`

Reason:

- The predecessor localized the current post-theta/QV `P/MU/W` residual to
  WRF `after_first_rk_step_part1` versus JAX `_physics_step_forcing.carry.state`.
- This sprint should record the narrower internal WRF boundary around
  `phy_prep` / `calc_p_rho_phi`, a narrow fix, or the exact missing truth
  contract.

Evidence:

- Opening evidence:
  `proofs/v014/step1_p_ph_mu_boundary_localization.json`.
- Closing evidence expected:
  `proofs/v014/step1_first_rk_part1_p_state_split.json`.

Proposed Destination:

- `.agent/memory/pending/2026-06-09-v014-step1-first-rk-part1-p-state-split.md`

Expected memory after close:

- Exact first internal WRF/JAX boundary for the current `P/MU/W` residual, or
  exact blocker.
- Whether any production source fix was made and its before/after top
  residuals.
- Whether the next debug step remains inside physics/source construction,
  pressure refresh, JAX adapter/carry construction, or another named boundary.
