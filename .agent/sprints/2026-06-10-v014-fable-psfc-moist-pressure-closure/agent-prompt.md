You are Fable high, debugging/fix worker for wrf_gpu2 v0.14.

Read, in order:
1. `PROJECT_CONSTITUTION.md`
2. `AGENTS.md`
3. `.agent/skills/managing-sprints/SKILL.md`
4. `.agent/sprints/2026-06-10-v014-fable-psfc-moist-pressure-closure/sprint-contract.md`
5. `.agent/reviews/2026-06-10-v014-gpt-psfc-vapor-light-analysis.md`

Task: close the fixed-Canary `PSFC` vapor-light pressure-state residual as one
whole endpoint-defined task. Implement a WRF-faithful fix if proven safe, or
return a precise WRF-anchored proof that it is not fixable within v0.14 scope.

Constraints:
- CPU-only unless the manager explicitly grants GPU.
- Do not touch unrelated files.
- No comparator-only tolerance relaxation.
- No output-only PSFC clamp unless WRF source proves that exact diagnostic path.
- Keep the final report compact and manager-actionable.

Current fixed run root:
`/mnt/data/wrf_gpu_validation/v014_canary_d02_72h_lbcfix_20260610T151455Z`

CPU truth:
`/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z`

WRF reference source:
`/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF`

Required outputs:
- `proofs/v014/psfc_moist_pressure_state_closure.md`
- `proofs/v014/psfc_moist_pressure_state_closure.json`
- `.agent/reviews/2026-06-10-v014-fable-psfc-moist-pressure-closure.md`

If you make source changes, run focused CPU tests/proofs and record exact
commands. When done, notify the manager pane with delayed repeated Enters:

```bash
tmux send-keys -t 0:2 'FABLE PSFC_MOIST_PRESSURE_CLOSURE DONE - see .agent/reviews/2026-06-10-v014-fable-psfc-moist-pressure-closure.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
