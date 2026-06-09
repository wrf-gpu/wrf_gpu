# Worker Report

Summary: produced a CPU-WRF source-derived post-RK refresh localization proof
for the h10 `d02` same-state grid-parity investigation. The sprint found a green
WRF compare surface and did not edit repo `src/`, use GPU, run TOST, or touch
the pristine WRF tree in place.

Files changed:

- `proofs/v014/wrf_post_rk_refresh_localization.py`
- `proofs/v014/wrf_post_rk_refresh_localization.json`
- `proofs/v014/wrf_post_rk_refresh_localization.md`
- `proofs/v014/wrf_post_rk_refresh_localization_patch.diff`
- `.agent/reviews/2026-06-09-v014-post-rk-refresh-localization.md`

Proof summary: verdict
`REFRESH_LAYER_GREEN_post_after_all_rk_steps_pre_halo`. Final
`calc_p_rho_phi` closes the large `P` gap after `small_step_finish`; the state
immediately after `after_all_rk_steps` before RK halo exchanges closes `V/W` and
matches the green marker/CPU h10 at exact or roundoff level.

Commands reported by worker:

- compile disposable WRF with `tcsh ./compile em_real`
- CPU WRF `mpirun --oversubscribe -np 28 ./wrf.exe`
- `python -m py_compile proofs/v014/wrf_post_rk_refresh_localization.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/wrf_post_rk_refresh_localization.py`
- `python -m json.tool proofs/v014/wrf_post_rk_refresh_localization.json >/tmp/wrf_post_rk_refresh_localization.validated.json`

Next recommendation: implement a JAX CPU same-state wrapper targeting
`post dyn_em/solve_em.F::after_all_rk_steps state before RK halo exchanges`.
