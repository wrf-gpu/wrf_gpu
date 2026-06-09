# Memory Patch Proposal: V0.14 Step-1 Live-Nest Perturb-State Init

Date: 2026-06-09 19:36 WEST

Reviewer Status: Pending. Sprint closed, proof validated by manager; do not
apply to stable memory until reviewer approval.

Scope:

- `.agent/memory/pending/2026-06-09-v014-step1-live-nest-perturb-state-init.md`

Reason:

- The sprint localized the current `P/MU/W` residual to WRF live-nest
  `start_domain(nest,.TRUE.)` perturbation-state initialization after
  base/theta/QV correction and before WRF `first_rk_step_part1`.
- Formula transcriptions show the family split: W is closed proof-locally, MU is
  near-closed through `press_adj`, and P still needs one internal WRF
  `al/alt`/pre-post-`press_adj` truth surface before a source patch.

Evidence:

- Opening evidence:
  `proofs/v014/step1_first_rk_part1_p_state_split.json`.
- Closing evidence:
  `proofs/v014/step1_live_nest_perturb_state_init.json`.
- Manager closeout:
  `.agent/sprints/2026-06-09-v014-step1-live-nest-perturb-state-init/manager-closeout.md`.

Proposed Destination:

- `.agent/memory/pending/2026-06-09-v014-step1-live-nest-perturb-state-init.md`

Proposed memory:

- WRF live-nest `P/MU/W` perturbation-state initialization is the active Step-1
  boundary.
- Current JAX leaves raw `wrfinput_d02` `P/MU/W` through raw child, live child,
  boundary package, initial carry, halo entry, and `_physics_step_forcing`.
- Proof-local residual reductions are `P_STATE` `69.96875 -> 3.9458582235092763`
  Pa, `MU_STATE` `13.256103515625 -> 0.047773029698646496` Pa, and `W_STATE`
  `0.7605466246604919 -> 1.2992081932505783e-07` m/s.
- No source fix yet: safe patching requires WRF `start_domain` surfaces after
  the hypsometric `P/al/alt` recompute and before/after `press_adj`.
