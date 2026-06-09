# Reviewer Report: V0.14 Step-1 Current-MUB/Base-Input Split

Date: 2026-06-09

Decision: ACCEPT as a proof sprint; require a separate source-changing sprint
for the transient live-nest adjust-base path.

## Findings

- HIGH: The proof explains the mismatch without a broad rewrite. WRF
  `adjust_tempqv` consumes current `MUB` immediately after `blend_terrain`,
  before the later `start_domain` base recompute. The JAX theta proof used the
  final post-`start_domain` base field for that earlier call.
- HIGH: The proof-side WRF blend reproduces the WRF adjust hook at the target
  cell within `4.6e-4 Pa`, while the final base field matches WRF pre-part1
  within `4.7e-3 Pa`. That cleanly separates the transient and final surfaces.
- MEDIUM: The sprint contract's grouped pressure formula was corrected. WRF
  source uses `p_new = p + c4h + c3h*mub + p_top`, not
  `p + c3h*(mub+p_top) + c4h`.
- LOW: The worker could not write the fresh `/mnt/data` scratch hook under its
  sandbox. The proposed patch diff is useful, but production patch acceptance
  still needs a field/full-domain validation surface.

## Evidence

- `proofs/v014/step1_current_mub_base_input_split.md`
- `proofs/v014/step1_current_mub_base_input_split.json`
- `proofs/v014/step1_current_mub_base_input_split_wrf_patch.diff`
- `.agent/reviews/2026-06-09-v014-step1-current-mub-base-input-split.md`

## Required Next Sprint

Open a source-changing sprint with a narrow write scope in the live-nest
initialization/theta-semantics path. It must compute the WRF post-blend /
pre-`start_domain` current `MUB` for the `adjust_tempqv` call, use it only for
theta/QV adjustment, keep final BaseState from `start_domain`, and rerun the
Step-1 theta/QV proof plus a field-level guard.
