# Manager Closeout: V0.14 Step-1 Current-MUB/Base-Input Split

Date: 2026-06-09 17:32 WEST

## Outcome

The sprint is closed with verdict
`STEP1_CURRENT_MUB_BASE_SPLIT_WRF_BLEND_UNIMPLEMENTED_OR_MISMATCHED`.

The `17.5 Pa` mismatch is a real boundary split. WRF `adjust_tempqv` uses a
transient post-`blend_terrain`/pre-`start_domain` current `MUB`; the prior JAX
theta proof used the final post-`start_domain` base `MUB`.

## Proof Objects

- `proofs/v014/step1_current_mub_base_input_split.py`
- `proofs/v014/step1_current_mub_base_input_split.json`
- `proofs/v014/step1_current_mub_base_input_split.md`
- `proofs/v014/step1_current_mub_base_input_split_wrf_patch.diff`
- `.agent/reviews/2026-06-09-v014-step1-current-mub-base-input-split.md`

Key facts:

- WRF adjust hook current `MUB`: `86812.25`
- JAX proof final base `MUB`: `86794.574960128695`
- Proof-side direct WRF blend `MUB`: `86812.250452109511`
- WRF pre-part1 final `MUB`: `86794.5703125`
- WRF source formula verified:
  `p_new = p + c4h + c3h*mub + p_top`

## Merge Decision:

Commit and push proof/review/sprint documentation only. No production source
change is made in this sprint. The next sprint may change source, but only
under a narrow contract that adds the transient live-nest adjust-base field and
reruns the Step-1 theta/QV proof.

## Scope Changes

The worker could not write `/mnt/data` scratch files from its sandbox, so it
did not run a fresh WRF hook. It recovered accepted scalar WRF truth from the
prior `adjust_tempqv` hook and provided a proposed disposable WRF patch diff.
That is sufficient for boundary classification, not for final source-patch
acceptance.

## Lessons

Do not use the final post-`start_domain` live-nest BaseState as the current base
input to WRF `adjust_tempqv`. The init path has two legitimate surfaces:
transient post-blend current `MUB` for theta/QV adjustment, and final
post-`start_domain` base fields for the later step-entry state.

## Next Sprint

Open `v014-step1-transient-adjust-base-fix`: implement the smallest production
or proof-side helper needed to compute transient post-blend/pre-`start_domain`
`MUB` for `adjust_tempqv`, keep the final BaseState unchanged, and rerun the
Step-1 theta/QV proof with field-level guards.
