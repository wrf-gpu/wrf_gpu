# V0.14 Step-1 T/P Operator Localization

Date: 2026-06-09

Verdict:
`STEP1_TP_LOCALIZED_RK_STAGE_ENTRY_STATE_AFTER_FIRST_RK_PARTS_RK1_T_STATE`.

`proofs/v014/step1_t_p_operator_localization.*` built a focused Step-1
substage WRF-vs-JAX comparator. It consumed 168 raw WRF truth files from
`/mnt/data/wrf_gpu2/v014_step1_t_p_operator_localization/wrf_truth`, emitted by
scratch-only env-gated WRF instrumentation documented in
`proofs/v014/step1_t_p_operator_localization_wrf_patch.diff`.

Key result:

- First strict and first material T/P-family mismatch:
  `after_rk_addtend_before_small_step_prep`, RK1, `T_STATE`.
- Largest material residual at that boundary:
  `PH_TEND` max_abs `794096.1875`, RMSE `26657.670278603327`.
- Other large boundary residuals include `RW_TEND`, `PH_TENDF`, `T_TEND`, and
  `T_TENDF`.
- RK1 `after_small_step_prep_calc_p_rho` work arrays match for `T_WORK` and
  `P_WORK` with max_abs `0.0`.
- Final Step-1 strict comparison still diverges: first `T`, top `P` max_abs
  `1561.2503728885986`.

Do not continue acoustic or final pressure-refresh debugging until the RK1
stage-entry source boundary is split. Next sprint should compare WRF
`first_rk_step_part1/part2` outputs and dry `*_tendf` fields against JAX
`_physics_step_forcing` output and pre-`small_step_prep` carry/state.
