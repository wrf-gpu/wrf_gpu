# Manager Closeout

Merge Decision: accept and land proof artifacts.

Objective: localize the exact pressure/rho/post-RK refresh cadence that bridges
Ptolemy's tile-local `small_step_finish` layer to Herschel's green post-RK
marker. The sprint completed this and found the next compare target.

Accepted verdict: `REFRESH_LAYER_GREEN_post_after_all_rk_steps_pre_halo`.

Important result: final `calc_p_rho_phi` closes `P`, while the state
immediately after `dyn_em/solve_em.F::after_all_rk_steps` and before RK halo
exchanges closes `V/W` and matches CPU h10 at exact or roundoff level for
`T/P/PB/U/V/W/PH/MU/MUB`. Retained GPU/JAX h10 remains divergent on that same
patch, so the next sprint is a JAX CPU same-state wrapper at this surface.

Manager validation:

- `python -m json.tool proofs/v014/wrf_post_rk_refresh_localization.json >/tmp/wrf_post_rk_refresh_localization.manager.validated.json`
- `python -m py_compile proofs/v014/wrf_post_rk_refresh_localization.py`

Roadmap effect: update `PROJECT_PLAN.md`,
`.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`, and pending memory to
name `post after_all_rk_steps pre-halo` as the current green WRF compare target.
