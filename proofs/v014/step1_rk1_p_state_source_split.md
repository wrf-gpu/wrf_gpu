# V0.14 Step-1 RK1 P_STATE Source Split

Verdict: `STEP1_RK1_P_STATE_SOURCE_REFUTED_STALE_PROOF_LOADER_BYPASS_NEXT_T_TENDF`.

## Result

- CPU backend: `cpu`; GPU used: `False`.
- RK1 `P_STATE` at `after_rk_addtend_before_small_step_prep`: stale proof loader max_abs `69.96875` Pa; patched Mythos init max_abs `0.0390625` Pa; gate `1.0` Pa.
- The material `P_STATE` source hypothesis is refuted for current production init. The comparator path was bypassing the Mythos `start_domain` perturbation init.
- `P_STATE/MU_STATE/W_STATE/PH_STATE` are below material gates through `after_first_rk_step_part1`, `after_first_rk_step_part2`, and RK1 `after_rk_addtend_before_small_step_prep` with the patched capture.
- RK1 `T_WORK/P_WORK/PH_WORK/MU_WORK/W_WORK` are exact at `small_step_prep/calc_p_rho(step=0)`.

## Remaining Boundary

- First material after P closes: `T_TENDF` at WRF `after_first_rk_step_part2`; `T_TEND` at RK1 `after_rk_addtend_before_small_step_prep`.
- Largest remaining tendency residuals at RK1 addtend: `PH_TEND` max_abs `794096.1875`, `RW_TEND` max_abs `131390.78717448562`, `PH_TENDF` max_abs `27082.453125`, `T_TEND` max_abs `6637.764233094031`, `T_TENDF` max_abs `2596.305908203125`.
- Empty-dry and full-dry JAX augment comparisons give the same huge tendency residuals, so fixed physics source leaves are not the cause of this P_STATE issue.

## Next

Split WRF first_rk_step_part2 T_TENDF and then RK1 after_rk_addtend T_TEND/PH_TEND/RW_TEND against JAX compute_advection_tendencies/_augment_large_step_tendencies with a patched-init capture. Do not enter acoustic substeps for this P_STATE issue; P_STATE is below material gate before small_step_prep.

Detailed metrics are in `proofs/v014/step1_rk1_p_state_source_split.json`.
