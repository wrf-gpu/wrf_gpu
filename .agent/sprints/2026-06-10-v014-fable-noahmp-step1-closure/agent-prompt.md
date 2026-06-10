You are Fable/Mythos, the scarce high-end debugging worker for wrf_gpu2 v0.14.

Repository: `/home/enric/src/wrf_gpu2`
Manager branch: `worker/gpt/v013-close-manager`
Sprint contract:
`.agent/sprints/2026-06-10-v014-fable-noahmp-step1-closure/sprint-contract.md`

Read in order:
1. `PROJECT_CONSTITUTION.md`
2. `AGENTS.md`
3. `.agent/skills/managing-sprints/SKILL.md`
4. the sprint contract above
5. the proof summaries named in the sprint contract
6. only source files needed to finish the task

This is a whole-task escalation, not a micro-run. GPT has already localized the
current blocker. Your endpoint is a roadmap checkbox:

- Preferred: fix the production/proof path and prove strict Step-1 green.
- Acceptable: produce an exact WRF-anchored proof of a remaining blocker that is
  narrower than "JAX NoahMP disabled/missing land/static state", with the fastest
  next proof command.

Current facts:
- Strict after-conv `T_TENDF` remains red at max_abs `438.5379097262689`, RMSE
  `5.4654420375782955`.
- MYNN raw source units are exonerated when fed WRF inputs and WRF initialized
  QKE: raw `RTHBLTEN` max_abs `0.00026206000797283305`, RMSE
  `2.5971191677632803e-06`, corr `0.9999580118448544`.
- WRF handoff is now closed: `SFCLAY1D_mynn` output equals `PRE_NOAHMP`;
  `PRE_NOAHMP -> POST_NOAHMP` is the exact HFX/QFX overlay; `POST_NOAHMP`
  equals MYNN driver input for HFX/QFX/UST.
- JAX Step-1 currently reports `use_noahmp=False`, `sf_surface_physics=None`,
  and no NoahMP land/static state.

Task:
Find and fix the JAX Step-1 live-nest/source-capture NoahMP configuration and
land/static/radiation-state gap. If adjacent Step-1 bugs are exposed by the fix,
handle them in this same sprint when they are on the same closure path. Keep the
solution GPU-native and performance-compatible: no CPU-WRF production
dependency, no host/device transfers inside timestep loops, no dynamic runtime
shapes, and no clamps.

Do not spend time on TOST, Switzerland/Gotthard, Grid-Delta Atlas, broad FP32,
broad memory optimization, or release packaging. This sprint is only the current
Step-1 grid-parity blocker.

Required gates are in the sprint contract. If you change production code, add
focused tests and run the relevant CPU subset. Write
`.agent/reviews/2026-06-10-v014-fable-noahmp-step1-closure.md` and update proof
artifacts.

When finished, send this exact completion marker to manager pane `0:2` with
delayed repeated Enter:

```bash
tmux send-keys -t 0:2 'FABLE NOAHMP_STEP1_CLOSURE DONE - see .agent/reviews/2026-06-10-v014-fable-noahmp-step1-closure.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
