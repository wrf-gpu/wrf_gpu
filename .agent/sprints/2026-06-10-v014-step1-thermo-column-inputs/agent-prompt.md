You are GPT-5.5 xhigh, debugger/fixer for wrf_gpu2 v0.14.

Repo: `/home/enric/src/wrf_gpu2`
Branch: `worker/gpt/v013-close-manager`
Base commit: `cdfdbbc2 v014 fix tsk znt sfclay sourcing`

Read first:

1. `PROJECT_CONSTITUTION.md`
2. `AGENTS.md`
3. `.agent/skills/managing-sprints/SKILL.md`
4. `.agent/sprints/2026-06-10-v014-step1-thermo-column-inputs/sprint-contract.md`
5. `proofs/v014/step1_tsk_znt_sourcing_fix.md`
6. `.agent/reviews/2026-06-10-v014-step1-tsk-znt-sourcing.md`
7. `src/gpuwrf/coupling/physics_couplers.py::_surface_column_view`

Mission:

Close, or reduce to one strictly narrower WRF-anchored blocker, the remaining
Step-1 `sfclay_mynn` input mismatch in `th_phy/t_phy/p_phy/dz8w`.

Important facts:

- MYNN cold-start QKE is fixed and the MYNN kernel is exonerated with WRF inputs.
- MYNN surface first-call semantics are fixed.
- TSK/ZNT/MAVAIL are fixed at the exact `sfclay_mynn` input hook:
  `TSK` max_abs `0.0`, `ZNT` max_abs `1.1920928910669204e-08`.
- Current blocker:
  `th_phy(kts)` max_abs `5.490148027499686 K`,
  derived `t_phy(kts)` max_abs `5.521345498302992 K`,
  `p_phy(kts)` max_abs `292.8203125 Pa`,
  while `u/v/qv(kts)` are near roundoff.
- Strict Step-1 remains red: max_abs `1497.6112467075195`, RMSE
  `13.252694871222973`.
- Do not run TOST, Switzerland, broad FP32, broad memory, Hermes, or Fable.
- CPU-only unless the manager later grants a short GPU probe.

Work style:

- Use the existing exact WRF hook outputs from
  `proofs/v014/step1_tsk_znt_sourcing_fix.py` first. Add a smaller hook only if
  it removes ambiguity.
- Think like an expert runtime debugger: freeze the boundary, compare exact
  arrays, rule out orientation/indexing before changing production code.
- If the local hypothesis is wrong, do not stop at "surface differs"; name the
  next exact blocker and why.
- If a performance-compatible local fix is proven, implement it.
- Keep top-level reports compact.

Deliver:

- `proofs/v014/step1_thermo_column_inputs.py`
- `proofs/v014/step1_thermo_column_inputs.json`
- `proofs/v014/step1_thermo_column_inputs.md`
- `proofs/v014/step1_thermo_column_inputs_wrf_patch.diff` if you change/add WRF
  hook code.
- `.agent/reviews/2026-06-10-v014-step1-thermo-column-inputs.md`
- focused tests if production code changes.

Run the acceptance gates from the sprint contract. If a gate is blocked, record
the exact blocker and fastest next command.

When done, print exactly:

`GPT STEP1_THERMO_COLUMN_INPUTS DONE - see proofs/v014/step1_thermo_column_inputs.md`

Then notify manager pane:

```bash
tmux send-keys -t 0:2 'GPT STEP1_THERMO_COLUMN_INPUTS DONE - see proofs/v014/step1_thermo_column_inputs.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
