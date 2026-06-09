# Reviewer Report: V0.14 Step-1 Live-Nest Perturbation-State Init

Decision: accept as a localization sprint, not a source-fix sprint.

review verdict: accept as a localization sprint.

evidence checked:

- `proofs/v014/step1_live_nest_perturb_state_init.md`
- `proofs/v014/step1_live_nest_perturb_state_init.json`
- `.agent/reviews/2026-06-09-v014-step1-live-nest-perturb-state-init.md`
- manager rerun of compile, proof execution, JSON validation, `git diff --
  src/gpuwrf`, and `git diff --check`

finding:

The proof supports the manager hypothesis but does not overclaim a source fix.
`W_STATE` is proof-locally closed by `set_w_surface`; `MU_STATE` is near-closed
by `press_adj`; `P_STATE` remains too far from exact for patching without one
more WRF `start_domain` internal truth surface.

required next gate:

Before source edits, emit WRF surfaces after the hypsometric `P/al/alt`
recompute and immediately before/after `press_adj`.
