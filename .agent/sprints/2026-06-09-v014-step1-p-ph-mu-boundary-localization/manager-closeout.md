# Manager Closeout: V0.14 Step-1 P/PH/MU Boundary Localization

Date: 2026-06-09 18:50 WEST

## Outcome

The sprint is closed with verdict
`STEP1_P_PH_MU_LOCALIZED_FIRST_RK_STEP_PART1_P_STATE`.

The proof localizes the current post-theta/QV P-family residual earlier than
boundary application, small-step prep, or `calc_p_rho(step=0)`: WRF
`after_first_rk_step_part1` versus JAX `_physics_step_forcing.carry.state`,
field `P_STATE`, max_abs `69.96875`.

## Proof Objects

- `proofs/v014/step1_p_ph_mu_boundary_localization.py`
- `proofs/v014/step1_p_ph_mu_boundary_localization.json`
- `proofs/v014/step1_p_ph_mu_boundary_localization.md`
- `.agent/reviews/2026-06-09-v014-step1-p-ph-mu-boundary-localization.md`

Key proof metrics:

- final `P` max_abs: `974.9820434775493`
- final `PH` max_abs: `67.3623167023926`
- final `MU` max_abs: `14.125275642998986`
- final `W` max_abs: `2.640715693903735`
- earliest checked `P_STATE` max_abs after `first_rk_step_part1`: `69.96875`
- earliest checked `MU_STATE` max_abs after `first_rk_step_part1`: `13.256103515625`
- earliest checked `W_STATE` max_abs after `first_rk_step_part1`: `0.7605466246604919`

## Merge Decision:

Commit and push the proof artifacts. Do not apply a source fix from this sprint:
the next boundary needs one narrower WRF scratch surface inside
`first_rk_step_part1` or a post-acoustic/pre-refresh pressure split.

## Validation

Manager reran:

- `python -m py_compile proofs/v014/step1_p_ph_mu_boundary_localization.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_p_ph_mu_boundary_localization.py`
- `python -m json.tool proofs/v014/step1_p_ph_mu_boundary_localization.json >/tmp/step1_p_ph_mu_boundary_localization.manager.validated.json`
- `git diff -- src/gpuwrf` with no output

## Scope Changes

No production source, GPU validation, TOST, Switzerland, FP32 source work,
memory source work, or Hermes was used.

## Next Sprint

Open `v014-step1-first-rk-part1-p-state-split`: instrument WRF inside
`first_rk_step_part1` around `phy_prep`/`calc_p_rho_phi` state writes for
`P/MU/W` and compare against the current JAX `_physics_step_forcing` boundary.
Only after that surface names a source bug should a source fix be attempted.
