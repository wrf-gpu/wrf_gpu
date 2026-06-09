# Worker Report: V0.14 Step-1 P/PH/MU Boundary Localization

Summary:

GPT-5.5 xhigh completed the focused CPU-only boundary/substage proof after the
Opus worker could not start due to Claude session limits. The proof reuses the
existing WRF source-boundary and T/P substage truth surfaces against the current
post-theta/QV JAX state.

Files Changed:

- `proofs/v014/step1_p_ph_mu_boundary_localization.py`
- `proofs/v014/step1_p_ph_mu_boundary_localization.json`
- `proofs/v014/step1_p_ph_mu_boundary_localization.md`
- `.agent/reviews/2026-06-09-v014-step1-p-ph-mu-boundary-localization.md`

Commands Run:

- `python -m py_compile proofs/v014/step1_p_ph_mu_boundary_localization.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_p_ph_mu_boundary_localization.py`
- `python -m json.tool proofs/v014/step1_p_ph_mu_boundary_localization.json >/tmp/step1_p_ph_mu_boundary_localization.validated.json`
- `git diff -- src/gpuwrf`

Proof Objects:

- `proofs/v014/step1_p_ph_mu_boundary_localization.json`
- `proofs/v014/step1_p_ph_mu_boundary_localization.md`
- `.agent/reviews/2026-06-09-v014-step1-p-ph-mu-boundary-localization.md`

Result:

Verdict is `STEP1_P_PH_MU_LOCALIZED_FIRST_RK_STEP_PART1_P_STATE`.

Key facts:

- Current final residuals remain `P` max_abs `974.9820434775493`, `PH`
  `67.3623167023926`, `MU` `14.125275642998986`, `W`
  `2.640715693903735`, `U` `0.7835467705023085`.
- First current material P-family state residual is WRF
  `after_first_rk_step_part1` versus JAX `_physics_step_forcing.carry.state`,
  field `P_STATE`, max_abs `69.96875`.
- `MU_STATE` and `W_STATE` are material at that same first checked boundary.
- RK1 `small_step_prep`/`calc_p_rho(step=0)` work arrays are exact for
  `T_WORK/P_WORK/PH_WORK/MU_WORK/W_WORK`.
- No production source fix was applied.

Handoff:

The next useful sprint should not edit dycore/source yet. It should emit one
new WRF scratch surface inside `first_rk_step_part1` around
`phy_prep`/`calc_p_rho_phi` state writes for `P/MU/W`, or split
post-acoustic/pre-refresh pressure before any source edit.
