You are GPT-5.5 xhigh, debugging worker for wrf_gpu2 v0.14.

Repository: `/home/enric/src/wrf_gpu2`
Branch/base: `worker/gpt/v013-close-manager`, commit `fc3c9fd9 v014 bound sfclay output algebra`
Sprint contract: `.agent/sprints/2026-06-10-v014-step1-mynn-source-coupling/sprint-contract.md`

Read in order:

1. `PROJECT_CONSTITUTION.md`
2. `AGENTS.md`
3. `.agent/skills/managing-sprints/SKILL.md`
4. the sprint contract above
5. only the prior proof summaries and source files needed for this task

Objective: close, or strictly narrow, the remaining Step-1 dry source divergence
after `SFCLAY1D_mynn` output algebra was bounded. Leading hypothesis:
MYNN/PBL source coupling after fixed surface outputs. Treat that hypothesis as
falsifiable. If it is wrong, use the context you already built to rank and test
the next hypotheses instead of returning only "not it".

Key prior result:

- `proofs/v014/step1_sfclay_output_algebra.md` verdict:
  `SFCLAY_OUTPUT_ALGEBRA_BOUNDED_NEXT_BLOCKER_MYNN_SOURCE_COUPLING`.
- Surface outputs are now small-bounded, but strict Step-1 `T_TENDF` remains
  red at max_abs `847.1446969755725`, RMSE `9.627208432391289`.

Required endpoint:

- Preferred: production fix plus proof that strict Step-1 `T_TENDF` is within
  max_abs `1.0e-3`, RMSE `1.0e-5`.
- Acceptable: exact WRF-anchored narrower blocker later/narrower than MYNN
  source coupling, with raw WRF MYNNEDMF input/source evidence and fastest next
  command.

Rules:

- CPU-only. Do not use GPU, Hermes, or Fable/Mythos.
- Do not touch memory/FP32, TOST, Switzerland/demo validation, release
  packaging, or broad docs.
- Allowed production files are only the ones in the sprint contract.
- Keep the implementation GPU-native and performance-compatible: no host/device
  transfers in timestep loops, no CPU fallback, no dynamic-shape runtime arrays,
  no correctness clamps.
- If production code changes, add focused tests and run them.
- Write all proof objects and a concise review report listed in the contract.

When finished, send this exact completion marker to manager pane `0:2` with
delayed repeated Enter:

```bash
tmux send-keys -t 0:2 'GPT STEP1_MYNN_SOURCE_COUPLING DONE - see proofs/v014/step1_mynn_source_coupling.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
