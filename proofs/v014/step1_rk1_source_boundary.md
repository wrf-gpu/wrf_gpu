# V0.14 Step-1 RK1 Source Boundary

Verdict: `STEP1_RK1_SOURCE_LOCALIZED_FIRST_RK_STEP_PART1_PHYSICS_STATE_MUTATION_T_STATE`.

## Result

- CPU backend: `cpu`.
- WRF source-boundary truth root: `<DATA_ROOT>/wrf_gpu2/v014_step1_rk1_source_boundary/wrf_truth`.
- Fastest rigorous method: `FOCUSED_STEP1_SOURCE_BOUNDARY_TRUTH_COMPARATOR_FASTEST_RIGOROUS_WALL_CLOCK`.
- First localized boundary: `after_first_rk_step_part1` field `T_STATE`; WRF vs JAX operational carry max_abs `5.490173101425171` rmse `1.9175184863907806`.
- Same WRF field vs `_physics_step_forcing.state` max_abs `5.490142455570492` rmse `1.9174736017582765`.
- Continuity check at prior pre-small-step boundary still shows top material `PH_TEND` max_abs `794096.1875`.
- RK1 `small_step_prep` continuity remains exact for `T_WORK` max_abs `0.0` and `P_WORK` max_abs `0.0`.

Detailed comparison tables are in `proofs/v014/step1_rk1_source_boundary.json`.
