You are GPT-5.5 xhigh, WRF instrumentation/debug worker for wrf_gpu2 v0.14.

Read:
- PROJECT_CONSTITUTION.md
- AGENTS.md
- .agent/skills/managing-sprints/SKILL.md
- .agent/sprints/2026-06-09-v014-step1-adjust-tempqv-intermediate/sprint-contract.md
- proofs/v014/step1_theta_same_qvapor.md
- proofs/v014/step1_theta_same_qvapor.json
- proofs/v014/step1_qvapor_precall_savepoint.md
- .agent/reviews/2026-06-09-v014-theta-qvapor-opus-critic.md

Task: emit or recover CPU-WRF exact adjust_tempqv intermediate values for the
Step-1 live-nest theta residual path. Target worst cell from the current proof:
zero {k:1,y:9,x:17}, Fortran {i:18,j:10,k:2}.

Do not edit src/gpuwrf/**. Do not use GPU. Do not run TOST, Switzerland, FP32
source work, or memory source work. Do not use Hermes.

Allowed scratch WRF tree:
/mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/WRF

Allowed scratch root:
/mnt/data/wrf_gpu2/v014_step1_adjust_tempqv_intermediate

Required deliverables:
- proofs/v014/step1_adjust_tempqv_intermediate.py
- proofs/v014/step1_adjust_tempqv_intermediate.json
- proofs/v014/step1_adjust_tempqv_intermediate.md
- proofs/v014/step1_adjust_tempqv_intermediate_wrf_patch.diff
- .agent/reviews/2026-06-09-v014-step1-adjust-tempqv-intermediate.md

Proof goal:
- Instrument disposable WRF, preferably around mediation_integrate.F /
  nest_init_utils.F::adjust_tempqv, to emit WRF pre/post t_2, QVAPOR, p_old,
  p_new, tc, rh, dth1, dth, p, pb, mub, mub_save, c3h, c4h, p_top for the
  target cell or a small neighborhood.
- Rebuild/run the shortest CPU-WRF truth capture possible. If sandboxed WRF run
  is blocked by OpenMPI/PMIx sockets, fail closed with exact command/log and say
  manager must rerun unsandboxed.
- Compare emitted WRF values to JAX proof values from
  proofs/v014/step1_theta_same_qvapor.json.
- Emit exactly one verdict from the sprint contract.

Validation:
python -m py_compile proofs/v014/step1_adjust_tempqv_intermediate.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_adjust_tempqv_intermediate.py
python -m json.tool proofs/v014/step1_adjust_tempqv_intermediate.json >/tmp/step1_adjust_tempqv_intermediate.validated.json
git diff -- src/gpuwrf

When done, notify manager pane exactly with delayed repeated Enter:
tmux send-keys -t 0:2 'GPT STEP1_ADJUST_TEMPQV_INTERMEDIATE DONE - see proofs/v014/step1_adjust_tempqv_intermediate.md' Enter; sleep 1; tmux send-keys -t 0:2 Enter; sleep 1; tmux send-keys -t 0:2 Enter
