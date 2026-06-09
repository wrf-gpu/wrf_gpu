# V0.14 Step-1 P/PH/MU Boundary Localization

Date: 2026-06-09 18:50 WEST

Sprint:
`.agent/sprints/2026-06-09-v014-step1-p-ph-mu-boundary-localization`.

Proof:
`proofs/v014/step1_p_ph_mu_boundary_localization.*`.

Verdict:
`STEP1_P_PH_MU_LOCALIZED_FIRST_RK_STEP_PART1_P_STATE`.

Important proof facts:

- Current post-theta/QV final Step-1 residuals remain material:
  - `P` max_abs `974.9820434775493`
  - `PH` max_abs `67.3623167023926`
  - `MU` max_abs `14.125275642998986`
  - `W` max_abs `2.640715693903735`
  - `U` max_abs `0.7835467705023085`
- First current material P-family state residual is WRF
  `after_first_rk_step_part1` versus JAX `_physics_step_forcing.carry.state`,
  field `P_STATE`, max_abs `69.96875`.
- `MU_STATE` and `W_STATE` are material at that same first checked boundary.
- RK1 `small_step_prep`/`calc_p_rho(step=0)` work arrays are exact for
  `T_WORK/P_WORK/PH_WORK/MU_WORK/W_WORK`.
- The old pre-theta-fix `T_STATE` source-boundary verdict is stale for this
  branch; the current rerun names `P_STATE`.
- No production source fix was made.

Manager conclusion:

Continue grid-parity debugging with an internal WRF `first_rk_step_part1`
surface around `phy_prep`/`calc_p_rho_phi` state writes for `P/MU/W`, or a
post-acoustic/pre-refresh pressure split if the next manager chooses the
downstream pressure path. Keep TOST, Switzerland, FP32 source work, and memory
follow-ups paused.
