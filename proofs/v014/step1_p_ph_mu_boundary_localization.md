# V0.14 Step-1 P/PH/MU Boundary Localization

Verdict: `STEP1_P_PH_MU_LOCALIZED_FIRST_RK_STEP_PART1_P_STATE`.

## Result

- CPU backend: `cpu`; GPU used: `False`.
- Required ancestor `3aa5f15b` present: `True`.
- Fastest rigorous method: `FOCUSED_STEP1_SOURCE_AND_SUBSTAGE_TRUTH_COMPARATOR_FASTEST_RIGOROUS_WALL_CLOCK`.
- Current post-theta/QV strict Step-1 final top residuals: `P` max_abs `974.9820434775493`, `PH` max_abs `67.3623167023926`, `MU` max_abs `14.125275642998986`, `W` max_abs `2.640715693903735`, `U` max_abs `0.7835467705023085`.
- First current material P-family state residual: WRF `after_first_rk_step_part1` vs JAX `_physics_step_forcing.carry.state`, field `P_STATE`, max_abs `69.96875`.
- `MU_STATE` and `W_STATE` are also material at that same first boundary; `PH_STATE` is not material there and first becomes material at RK2 stage entry.
- RK1 `small_step_prep`/`calc_p_rho(step=0)` work arrays are exact for `T_WORK/P_WORK/PH_WORK/MU_WORK/W_WORK`.
- No production source fix was applied.

## Field Table

| Field | Earliest checked material boundary | Internal | max_abs | RMSE | Worst boundary band |
|---|---|---:|---:|---:|:--:|
| T | rk2_after_rk_addtend_before_small_step_prep | T_STATE | 0.2004841503570276 | 0.002279166276077141 | True |
| P | wrf_after_first_rk_step_part1_vs_jax_physics_carry | P_STATE | 69.96875 | 1.161700383780558 | True |
| PH | rk2_after_rk_addtend_before_small_step_prep | PH_STATE | 14.646314212531024 | 3.547343936190633 | False |
| MU | wrf_after_first_rk_step_part1_vs_jax_physics_carry | MU_STATE | 13.256103515625 | 0.5968840909134501 | True |
| W | wrf_after_first_rk_step_part1_vs_jax_physics_carry | W_STATE | 0.7605466246604919 | 0.01470981557635803 | False |
| U | final_post_after_all_rk_steps_pre_halo | U | 0.7835467705023085 | 0.02056418486197857 | True |

## Distinctions

- Boundary package vs application: package leaf equality is not directly emitted by the existing WRF truth, but the first P/MU/W state residual is before WRF dry boundary application and before JAX `apply_lateral_boundaries`.
- RK source/tendency vs prep: state residuals pre-exist `rk_tendency/rk_addtend_dry`; RK1 prep/calc_p_rho work fields are exact.
- Pressure refresh vs acoustic finish: current surfaces do not split post-acoustic/pre-refresh from refreshed pressure, so no narrow fix is justified.
- Boundary band vs interior: final P worst cell is boundary-band, but the residuals are not boundary-only.
- Stale proofs: previous source/TP JSON verdicts named pre-theta-fix `T_STATE`; current rerun names `P_STATE`.

Detailed metrics are in `proofs/v014/step1_p_ph_mu_boundary_localization.json`.
