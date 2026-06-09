# V0.14 Post-RK Refresh Localization Review

- objective: localize the pressure/rho/post-RK refresh cadence between Ptolemy's post-`small_step_finish` layer and Herschel's green post marker.
- verdict: `REFRESH_LAYER_GREEN_post_after_all_rk_steps_pre_halo`.
- files changed: `proofs/v014/wrf_post_rk_refresh_localization.py`, `.json`, `.md`, `_patch.diff`, and this review.
- WRF copy/run paths: `/mnt/data/wrf_gpu2/v014_post_rk_refresh/WRF`, `/mnt/data/wrf_gpu2/v014_post_rk_refresh/run_case3`.
- patch hash: `fe5e0666961e4b0f106d6427b1dbdcbbc2edb78990e7b4943f7f36a299e60091`.
- executable hash: `9b830e844252c5dc70b858bdfd20cd3a99b84e5c4c7ffcb68607ef658cabcb0c`.
- proof objects: `/home/enric/src/wrf_gpu2/proofs/v014/wrf_post_rk_refresh_localization.json`, `/home/enric/src/wrf_gpu2/proofs/v014/wrf_post_rk_refresh_localization.md`, `/home/enric/src/wrf_gpu2/proofs/v014/wrf_post_rk_refresh_localization_patch.diff`.
- commands run: see JSON `commands`.
- unresolved risks: no JAX same-state wrapper; selected h10 patch only; retained GPU wrfout is not fresh.
- next decision needed: use `post dyn_em/solve_em.F::after_all_rk_steps state before RK halo exchanges` as the CPU wrapper target if accepted, otherwise instrument a narrower sub-boundary.
