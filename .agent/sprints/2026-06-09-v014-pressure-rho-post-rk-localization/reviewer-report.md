# Reviewer Report

Decision: accepted as the WRF-side green compare surface for the next
same-state JAX comparison.

The proof satisfies the contract: it bridges Ptolemy's `post_small_step_finish`
layer to the accepted post-RK marker, explicitly names the cadence boundary, and
does not claim a production root cause. The data show a two-step closure:
`calc_p_rho_phi` closes `P`, then `after_all_rk_steps` closes `V/W` before RK
halo exchanges.

Material evidence reviewed:

- `proofs/v014/wrf_post_rk_refresh_localization.md`
- `proofs/v014/wrf_post_rk_refresh_localization.json`
- `proofs/v014/wrf_post_rk_refresh_localization_patch.diff`
- `.agent/reviews/2026-06-09-v014-post-rk-refresh-localization.md`

Key retained risk: this is selected-patch h10 evidence, not full-grid or
full-column coverage, and no JAX wrapper has yet compared model internals at the
same surface. Those are follow-up scope, not defects in this sprint.

Required follow-up: JAX CPU same-state wrapper at
`post after_all_rk_steps pre-halo`, with WRF truth from this sprint.
