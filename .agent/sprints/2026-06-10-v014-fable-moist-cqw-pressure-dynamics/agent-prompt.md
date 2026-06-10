You are Fable/Mythos, hard-kernel debug/fix worker for wrf_gpu2 v0.14.

First read:
1. `PROJECT_CONSTITUTION.md`
2. `AGENTS.md`
3. `.agent/skills/managing-sprints/SKILL.md`
4. `.agent/sprints/2026-06-10-v014-fable-moist-cqw-pressure-dynamics/sprint-contract.md`
5. `.agent/reviews/2026-06-10-v014-fable-psfc-moist-pressure-closure.md`
6. `proofs/v014/psfc_moist_pressure_state_closure.md`
7. `proofs/v014/psfc_moist_pressure_gpu_h4_validation.md`

Task: close or formally bound the remaining 3D pressure-state dynamics blocker
as one whole endpoint-defined sprint. The accepted PSFC diagnostic fix is done;
do not re-litigate it unless you find a proof-level error. The suspected blocker
is operational dry `cqw` / `pg_buoy_w_dry` in the acoustic w-equation, causing
GPU `P+PB` to ride a dry hydrostatic column while CPU WRF rides a moist one.

Endpoint: implement a WRF-faithful moist `cqw` / moist `pg_buoy_w` path with
proofs, or return a formal WRF-anchored bound/impossibility proof and exact next
plan. Keep source changes minimal, GPU-native, and performance-safe. CPU proof
first. Use GPU only for a short focused gate if your CPU proof is strong and the
manager/GPU lock is clear.

Required outputs:
- `proofs/v014/moist_cqw_pressure_dynamics_closure.py`
- `proofs/v014/moist_cqw_pressure_dynamics_closure.json`
- `proofs/v014/moist_cqw_pressure_dynamics_closure.md`
- `.agent/reviews/2026-06-10-v014-fable-moist-cqw-pressure-dynamics.md`

When done, notify manager pane:

```bash
tmux send-keys -t 0:2 'FABLE MOIST_CQW_PRESSURE_DYNAMICS DONE - see .agent/reviews/2026-06-10-v014-fable-moist-cqw-pressure-dynamics.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
