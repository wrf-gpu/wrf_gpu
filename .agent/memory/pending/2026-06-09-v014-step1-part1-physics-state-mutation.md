# V0.14 Step-1 Part1 Physics-State Mutation

Date: 2026-06-09

Verdict:
`STEP1_PART1_INPUT_ALREADY_DIVERGED_T_STATE`.

`proofs/v014/step1_part1_physics_state_mutation.*` built an internal WRF
`first_rk_step_part1` savepoint/comparator using scratch-only, env-gated WRF
instrumentation documented in
`proofs/v014/step1_part1_physics_state_mutation_wrf_patch.diff`.

Key result:

- `part1_entry_before_init_zero_tendency` `T_STATE` vs JAX live-nest
  step-entry state: max_abs `5.490173101425171`, RMSE
  `1.9175184863907806`.
- Largest WRF internal `T_STATE` delta from part1 entry across instrumented
  part1 boundaries: max_abs `0.0`.
- `part1_exit` retains the same residual against JAX carry/state surfaces.

Do not continue debugging radiation/surface/PBL/cumulus or acoustic internals
for this `T_STATE` residual. The next sprint should move upstream to the
live-nest/WRF handoff immediately before `first_rk_step_part1` entry and compare
the WRF call-site state to JAX Step-1 loader/carry/state surfaces.
