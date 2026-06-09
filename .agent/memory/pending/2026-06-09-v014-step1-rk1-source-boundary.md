# V0.14 Step-1 RK1 Source Boundary

Date: 2026-06-09

Verdict:
`STEP1_RK1_SOURCE_LOCALIZED_FIRST_RK_STEP_PART1_PHYSICS_STATE_MUTATION_T_STATE`.

`proofs/v014/step1_rk1_source_boundary.*` built a focused Step-1
source-boundary comparator using scratch-only, env-gated WRF instrumentation
documented in `proofs/v014/step1_rk1_source_boundary_wrf_patch.diff`.

Key result:

- First localized source boundary: `after_first_rk_step_part1`.
- Field: `T_STATE`.
- WRF vs JAX operational carry max_abs `5.490173101425171`, RMSE
  `1.9175184863907806`.
- WRF vs `_physics_step_forcing.state` max_abs `5.490142455570492`, RMSE
  `1.9174736017582765`.
- RK1 `small_step_prep` continuity remains exact for `T_WORK` and `P_WORK`,
  both max_abs `0.0`.

Do not continue acoustic or final pressure-refresh debugging until the internal
WRF `first_rk_step_part1` mutation is split. The next sprint should instrument
or compare WRF `first_rk_step_part1` internals against JAX
`_physics_step_forcing` state/tendency output and operational carry.
