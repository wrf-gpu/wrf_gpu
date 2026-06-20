# V0.14 Step-1 Pre-Part1 Handoff

Verdict: `STEP1_PRE_PART1_LOCALIZED_JAX_LOADER_T_STATE`.

## Result

- CPU backend: `cpu`.
- WRF solve_em truth root: `<DATA_ROOT>/wrf_gpu2/v014_step1_pre_part1_handoff/wrf_truth`.
- WRF patch artifact: `<USER_HOME>/src/wrf_gpu2/proofs/v014/step1_pre_part1_handoff_wrf_patch.diff`.
- Full-vs-perturbation theta conclusion: `WRF_T_STATE_IS_PERTURBATION_THETA`.
- WRF `T_STATE` delta from `after_step_increment` to `before_first_rk_step_part1_call`: max_abs `0.0`, rmse `0.0`.
- WRF `before_first_rk_step_part1_call` `T_STATE` vs raw JAX live-nest input state (`State.theta - 300 K`): max_abs `5.490173101425171`, rmse `1.9175184863907806`.
- WRF `before_first_rk_step_part1_call` `T_STATE` vs JAX step-entry haloed state (`State.theta - 300 K`): max_abs `5.490173101425171`, rmse `1.9175184863907806`.
- WRF solve_em pre-call vs prior part1-entry `T_STATE` continuity: max_abs `0.0`.

## Interpretation

- WRF does not change `grid%t_2` between the solve_em step boundary and the `CALL first_rk_step_part1` call-site.
- The previous part1-entry surface is continuous with the new solve_em pre-call surface for `T_STATE`.
- The 300 K offset check rejects the full-theta mismatch explanation: WRF `grid%t_2` is perturbation theta and should be compared to JAX `State.theta - 300 K`.
- The residual is already present in the raw JAX live-nest Step-1 state/carry before `_physics_step_forcing`; localize the loader/carry construction next.

Detailed comparison tables are in `proofs/v014/step1_pre_part1_handoff.json`.
