You are GPT-5.5 xhigh, WRF savepoint/debug worker for wrf_gpu2 v0.14.

Read:
- PROJECT_CONSTITUTION.md
- AGENTS.md
- .agent/skills/managing-sprints/SKILL.md
- .agent/sprints/2026-06-09-v014-step1-qvapor-precall-savepoint/sprint-contract.md

Task:
Create the missing same-boundary WRF `QVAPOR` truth at
`before_first_rk_step_part1_call`. This is a disposable CPU-WRF debug hook
extension only. Do not edit `src/gpuwrf/**`. Do not use GPU. Do not run TOST,
Switzerland, FP32, or memory work. Do not use Hermes.

Key context:
- `proofs/v014/step1_live_nest_theta_semantics.json` reduced the T residual
  from 5.49 K to 0.0054 K with WRF theta_m + adjust_tempqv semantics, but did
  not authorize a production patch.
- `proofs/v014/step1_qvapor_precall_truth_schema.json` proves accepted pre-call
  QVAPOR truth is missing and existing QVAPOR truth is post-RK/pre-halo.
- Existing disposable hook:
  `/mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/WRF/dyn_em/solve_em.F`
  subroutine `wrfgpu2_dump_pre_part1_surface`.

Required deliverables:
- `proofs/v014/step1_qvapor_precall_savepoint.py`
- `proofs/v014/step1_qvapor_precall_savepoint.json`
- `proofs/v014/step1_qvapor_precall_savepoint.md`
- `proofs/v014/step1_qvapor_precall_savepoint_wrf_patch.diff`
- `.agent/reviews/2026-06-09-v014-step1-qvapor-precall-savepoint.md`

Validation:
Run at minimum:
`python -m py_compile proofs/v014/step1_qvapor_precall_savepoint.py`
`JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_qvapor_precall_savepoint.py`
`python -m json.tool proofs/v014/step1_qvapor_precall_savepoint.json >/tmp/step1_qvapor_precall_savepoint.validated.json`
`git diff -- src/gpuwrf`

Final output:
Write artifacts, then notify manager pane exactly with delayed repeated Enter:
`tmux send-keys -t 0:2 'GPT STEP1_QVAPOR_PRECALL_SAVEPOINT DONE - see proofs/v014/step1_qvapor_precall_savepoint.md' Enter; sleep 1; tmux send-keys -t 0:2 Enter; sleep 1; tmux send-keys -t 0:2 Enter`
