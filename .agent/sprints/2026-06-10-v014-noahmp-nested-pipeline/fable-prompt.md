You are Fable high, implementation/debug worker for wrf_gpu2 v0.14.

Worktree:
`/home/enric/src/wrf_gpu2/.claude/worktrees/fable-noahmp-nested`

Base requirement:
- Verify `git log -1 --oneline` is `7c819067 v014 fix moist cqw pressure dynamics`.
- Work only in that worktree for source/proof edits and commit on branch
  `worker/fable/v014-noahmp-nested`.

Sprint contract:
`/home/enric/src/wrf_gpu2/.agent/sprints/2026-06-10-v014-noahmp-nested-pipeline/sprint-contract.md`

Read and execute the contract. Important constraints:
- Do not use the GPU.
- Do not stop or touch the running Canary 72h GPU run.
- Fix the whole endpoint if safe: nested standalone pipeline must activate and
  seed Noah-MP per domain when the case namelist has `sf_surface_physics=4`.
- Produce proof objects and a worker report.
- Commit the source/proof changes in the worktree branch.

Primary context:
- Fable h24 residual review:
  `/home/enric/src/wrf_gpu2/.agent/reviews/2026-06-10-v014-fable-canary-h24-residual.md`
- Proof:
  `/home/enric/src/wrf_gpu2/proofs/v014/canary_h24_residual_adjudication.md`
- Existing Noah-MP wiring references:
  `src/gpuwrf/io/noahmp_land_init.py`,
  `src/gpuwrf/runtime/operational_mode.py`,
  `src/gpuwrf/runtime/operational_state.py`,
  `proofs/m20/tost_noahmp_runner.py`,
  `proofs/noahmp/s6b_activate_validate.py`,
  `proofs/v014/step1_live_nest_init_rerun.py`.

Deliverables:
- `.agent/sprints/2026-06-10-v014-noahmp-nested-pipeline/worker-report.md`
- `proofs/v014/noahmp_nested_pipeline_activation.{json,md}`
- Worker branch commit hash.

Completion marker to manager pane:
`FABLE NOAHMP_NESTED_PIPELINE DONE - see .agent/sprints/2026-06-10-v014-noahmp-nested-pipeline/worker-report.md`

Use delayed repeated Enter presses exactly as in the contract.
