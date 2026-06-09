# Memory Patch Proposal: V0.14 Step-1 First-RK Part1 P-State Split

Date: 2026-06-09 18:57 WEST

Reviewer Status: APPROVED_FOR_PENDING_MEMORY.

Scope:

- `.agent/memory/pending/v014-grid-parity.md`

Reason:

- The sprint proved WRF `before_first_rk_step_part1_call` to
  `after_first_rk_step_part1` is exact for `P_STATE/MU_STATE/W_STATE/PH_STATE`.
- The same residual is already present in JAX `raw_child_state` and remains
  unchanged through `live_child_state`, boundary package, initial carry, halo,
  and `_physics_step_forcing.carry.state`.
- Therefore `first_rk_step_part1`, `phy_prep`, carry, halo, and boundary package
  are not the first `P/MU/W` fault surface.

Evidence:

- `proofs/v014/step1_first_rk_part1_p_state_split.json`
- `proofs/v014/step1_first_rk_part1_p_state_split.md`
- `.agent/reviews/2026-06-09-v014-step1-first-rk-part1-p-state-split.md`

Proposed Destination:

- `.agent/memory/pending/2026-06-09-v014-step1-first-rk-part1-p-state-split.md`

Approved pending memory:

- Verdict:
  `STEP1_FIRST_RK_PART1_P_STATE_LOCALIZED_PRE_PART1_RAW_CHILD_STATE`.
- WRF pre-call to after-part1 exact deltas:
  `P_STATE/MU_STATE/W_STATE/PH_STATE = 0.0`.
- JAX `raw_child_state`, `live_child_state`, and `haloed_step_entry_state`
  versus WRF pre-call all retain the material residuals:
  `P_STATE=69.96875`, `MU_STATE=13.256103515625`,
  `W_STATE=0.7605466246604919`, `PH_STATE=0.00048828125`.
- No source fix was made.
- Next sprint should transcribe/prove WRF live-nest perturbation-state
  initialization for `P_STATE/MU_STATE/W_STATE` from raw child to pre-part1
  state, then patch only if a narrow GPU-native formula is proven.
