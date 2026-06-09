# V0.14 Step-1 Pre-Part1 Handoff

Date: 2026-06-09

Verdict:
`STEP1_PRE_PART1_LOCALIZED_JAX_LOADER_T_STATE`.

`proofs/v014/step1_pre_part1_handoff.*` built a WRF solve_em pre-part1
savepoint/comparator using scratch-only, env-gated WRF instrumentation
documented in `proofs/v014/step1_pre_part1_handoff_wrf_patch.diff`.

Key result:

- WRF `T_STATE` delta from `after_step_increment` to
  `before_first_rk_step_part1_call`: max_abs `0.0`.
- WRF solve_em pre-call vs prior part1-entry `T_STATE` continuity: max_abs
  `0.0`.
- WRF pre-call `T_STATE` vs raw JAX live-nest input state
  (`State.theta - 300 K`): max_abs `5.490173101425171`, RMSE
  `1.9175184863907806`.
- Full-vs-perturbation theta check concluded
  `WRF_T_STATE_IS_PERTURBATION_THETA`.

Do not continue WRF solve_em, `first_rk_step_part1`, physics, or acoustic
debugging for this `T_STATE` residual. The next sprint should split JAX
live-nest Step-1 loader/carry construction for `T_STATE` against WRF solve_em
pre-call truth.
