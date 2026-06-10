You are GPT-5.5 xhigh, debugging worker for wrf_gpu2 v0.14.

Repository: `/home/enric/src/wrf_gpu2`
Branch/base: `worker/gpt/v013-close-manager`, commit `919334c0 v014 narrow mynn source to land flux handoff`
Sprint contract: `.agent/sprints/2026-06-10-v014-step1-surface-land-flux-handoff/sprint-contract.md`

Read in order:
1. `PROJECT_CONSTITUTION.md`
2. `AGENTS.md`
3. `.agent/skills/managing-sprints/SKILL.md`
4. the sprint contract above
5. only the prior proof summaries and source files needed for this task

Objective:
Close, or strictly narrow, the Step-1 heat/moisture flux handoff divergence between WRF surface/land physics and the MYNN driver.

Known facts:
- `proofs/v014/step1_mynn_source_coupling.json` verdict is `STEP1_MYNN_SOURCE_COUPLING_NARROWED_TO_SURFACE_LAND_FLUX_HANDOFF`.
- Strict after-conv `T_TENDF` remains red: max_abs `438.5379097262689`, RMSE `5.4654420375782955`.
- WRF MYNN inputs + WRF initialized QKE exonerate MYNN raw source units: raw `RTHBLTEN` max_abs `0.00026206000797283305`, RMSE `2.5971191677632803e-06`, corr `0.9999580118448544`.
- WRF `SFCLAY1D_mynn` output -> WRF MYNN-driver input: UST is closed (`4.998779168374767e-12`) but HFX is not (`277.80298614281253`) and QFX is not (`1.4684322196e-05`).

Central hypothesis to test:
WRF applies a surface/land flux overlay between surface-layer output and MYNN input, likely near `module_surface_driver` / `sf_surface_physics=4`. This may be missing, misordered, or misconfigured in the JAX Step-1 path. Treat this as a hypothesis, not a fact; confirm the actual WRF path with hooks and namelist values.

Required endpoint:
- Preferred: production fix plus proof that strict Step-1 `T_TENDF` is within max_abs `1.0e-3`, RMSE `1.0e-5`.
- Acceptable: exact WRF-anchored narrower blocker later/narrower than surface/land flux handoff, with hook evidence and fastest next command.

Rules:
- CPU-only. Do not use GPU, Hermes, or Fable/Mythos.
- Do not touch dycore, memory/FP32, TOST, Switzerland/demo validation, release packaging, or broad docs.
- Allowed production files are only the ones in the sprint contract.
- Keep implementation GPU-native and performance-compatible: no host/device transfers in timestep loops, no CPU fallback, no dynamic-shape runtime arrays, no clamps.
- If production code changes, add focused tests and run them.
- Write all proof objects and a concise review report listed in the contract.

When finished, send this exact completion marker to manager pane `0:2` with delayed repeated Enter:

```bash
tmux send-keys -t 0:2 'GPT STEP1_SURFACE_LAND_FLUX_HANDOFF DONE - see proofs/v014/step1_surface_land_flux_handoff.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
