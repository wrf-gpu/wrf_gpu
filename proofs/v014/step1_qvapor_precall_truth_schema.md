# V0.14 Step-1 QVAPOR Pre-Call Truth Schema

Verdict: `STEP1_QVAPOR_PRECALL_TRUTH_MISSING_SAVEPOINT_SPEC_READY`.

## Result

- GPU used: `false`.
- Target WRF boundary: `before_first_rk_step_part1_call`, domain `2`, step `1`, rk `1`.
- Accepted pre-call text files: `28` under `/mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/wrf_truth`.
- Accepted pre-call mass schema: `['fortran_i', 'fortran_j', 'fortran_k', 'zero_x', 'zero_y', 'zero_k', 'T_STATE', 'P_STATE', 'PB', 'MU_STATE', 'MUB', 'MUT']`.
- Accepted pre-call W/PH schema: `None`.
- QVAPOR at accepted pre-call boundary: `False`.
- Existing QVAPOR truth artifact groups: `2`.

## QVAPOR Inventory

| Artifact | Boundary | Step/RK | Shape | Same-boundary? | Classification |
|---|---|---:|---|---:|---|
| 28 text tiles: `/mnt/data/wrf_gpu2/v014_step1_same_input_truth/raw_truth` | `post_after_all_rk_steps_pre_halo` | 1/4 | `[44, 66, 159]` | `no` | `post_rk_or_different_boundary` |
| 1 npz: `/mnt/data/wrf_gpu2/v014_same_input_contract_builder/wrf_truth/same_input_post_after_all_rk_steps_pre_halo_d02_step_1.npz` | `post_after_all_rk_steps_pre_halo` | 1/None | `[44, 66, 159]` | `no` | `post_rk_or_different_boundary` |

## Boundary Evidence

- The accepted pre-call hook is immediately before `CALL first_rk_step_part1` in the instrumented Step-1 WRF tree.
- The accepted dump schema writes `T_STATE/P_STATE/PB/MU_STATE/MUB/MUT` and `W_STATE/PH_STATE/PHB`, but not `QVAPOR`.
- The only QVAPOR-bearing Step-1 truth found is `post_after_all_rk_steps_pre_halo` with `rk_step 4`, plus the promoted NPZ from that same boundary.

## Theta Contract

- WRF `grid%t_2` is perturbation theta; with `use_theta_m=1`, `adjust_tempqv` stores moist-theta perturbation (`theta_m - 300`).
- For the next proof, same-boundary WRF `QVAPOR` is required to convert between WRF moist theta and any dry `State.theta` convention.
- If JAX `State.theta` remains dry full theta, compare it to `(WRF T_STATE + 300)/(1 + R_v/R_d * WRF QVAPOR)` and compare derived JAX `theta_m - 300` to WRF `T_STATE`.

## Next Action

Add a minimal CPU-WRF savepoint extension at the existing `before_first_rk_step_part1_call` hook to emit `QVAPOR` from `moist(i,k,j,P_QV)` on the mass grid, preserving all existing pre-call fields for identity checks. Do not reuse post-RK/pre-halo QVAPOR for the live-nest theta/debug proof.
