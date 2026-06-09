# V0.14 Step-1 T/P Operator Localization

Verdict: `STEP1_TP_LOCALIZED_RK_STAGE_ENTRY_STATE_AFTER_FIRST_RK_PARTS_RK1_T_STATE`.

## Result

- CPU backend: `cpu`.
- WRF substage truth root: `/mnt/data/wrf_gpu2/v014_step1_t_p_operator_localization/wrf_truth`.
- Fastest rigorous method: `FOCUSED_STEP1_SUBSTAGE_TRUTH_COMPARATOR_FASTEST_RIGOROUS_WALL_CLOCK`.
- First material T/P-family mismatch: `after_rk_addtend_before_small_step_prep` RK`1` field `T_STATE`.
- Top material residual there: `PH_TEND` max_abs `794096.1875` rmse `26657.670278603327`.
- RK`1` prep work arrays then match for `T_WORK` max_abs `0.0` and `P_WORK` max_abs `0.0`.
- Final accepted strict comparison still diverges: first `T`, top `P` max_abs `1561.2503728885986`.

Detailed per-boundary metrics are in `proofs/v014/step1_t_p_operator_localization.json`.
