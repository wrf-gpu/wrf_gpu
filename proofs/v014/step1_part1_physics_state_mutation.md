# V0.14 Step-1 Part1 Physics-State Mutation

Verdict: `STEP1_PART1_INPUT_ALREADY_DIVERGED_T_STATE`.

## Result

- CPU backend: `cpu`.
- WRF internal truth root: `<DATA_ROOT>/wrf_gpu2/v014_step1_part1_physics_state_mutation/wrf_truth`.
- Scratch WRF patch: `<USER_HOME>/src/wrf_gpu2/proofs/v014/step1_part1_physics_state_mutation_wrf_patch.diff`.
- Fastest rigorous method: `RIGHT_TOOL_FASTEST_WALL_CLOCK_SAVEPOINT_COMPARATOR`.
- `part1_entry_before_init_zero_tendency` `T_STATE` vs JAX live-nest step-entry state: max_abs `5.490173101425171`, rmse `1.9175184863907806`.
- Largest WRF internal `T_STATE` delta from part1 entry occurs at `after_init_zero_tendency`: max_abs `0.0`.
- `part1_exit` `T_STATE` vs JAX `_physics_step_forcing.carry.state`: max_abs `5.490173101425171`, rmse `1.9175184863907806`.
- `part1_exit` `T_STATE` vs JAX `_physics_step_forcing.state`: max_abs `5.490142455570492`, rmse `1.9174736017582765`.

## Interpretation

- The first material Step-1 `T_STATE` residual is already present at WRF `first_rk_step_part1` entry.
- The WRF routine itself does not materially mutate `grid%t_2` / `T_STATE` during the instrumented part1 boundaries.
- The next split should move upstream to the boundary immediately before this call, not into radiation/surface/PBL/cumulus leaves.

Detailed comparison tables are in `proofs/v014/step1_part1_physics_state_mutation.json`.
