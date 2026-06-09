You are Opus 4.8 xhigh, correctness-critical source worker for wrf_gpu2 v0.14.

Read:
- PROJECT_CONSTITUTION.md
- AGENTS.md
- .agent/skills/managing-sprints/SKILL.md
- .agent/sprints/2026-06-09-v014-step1-live-nest-theta-qv-wiring/sprint-contract.md
- proofs/v014/step1_transient_adjust_base_fix.md
- proofs/v014/step1_transient_adjust_base_fix.json
- proofs/v014/step1_qvapor_precall_savepoint.md
- src/gpuwrf/integration/d02_replay.py

Task: wire WRF theta_m conversion + adjust_tempqv into production live-nest
child initialization, using `_wrf_live_nest_transient_adjust_mub`. The prior
sprint proved the candidate closes theta, but production is not wired yet.

Allowed production source:
- src/gpuwrf/integration/d02_replay.py

Required proof artifacts:
- proofs/v014/step1_live_nest_theta_qv_wiring.py
- proofs/v014/step1_live_nest_theta_qv_wiring.json
- proofs/v014/step1_live_nest_theta_qv_wiring.md
- .agent/reviews/2026-06-09-v014-step1-live-nest-theta-qv-wiring.md

Optional proof updates:
- proofs/v014/step1_live_nest_init_rerun.py/json/md only if needed.

Do not edit unrelated production files. Do not use GPU. Do not run TOST,
Switzerland, FP32 source work, memory source work, or Hermes.

Proof requirements:
- Production live-nest init, not only proof-local candidate, applies WRF
  theta_m conversion + adjust_tempqv when appropriate.
- Transient adjust-base MUB matches WRF adjust hook.
- Final BaseState MUB remains matched to WRF pre-part1 final target.
- Theta and QVAPOR match same-boundary WRF pre-call truth.
- Run or strictly replace the Step-1 same-input 16-field comparison and name
  the next field/boundary if not fully closed.

Validation:
python -m py_compile src/gpuwrf/integration/d02_replay.py proofs/v014/step1_live_nest_theta_qv_wiring.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_live_nest_theta_qv_wiring.py
python -m json.tool proofs/v014/step1_live_nest_theta_qv_wiring.json >/tmp/step1_live_nest_theta_qv_wiring.validated.json
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src pytest -q tests/test_m7_l2_d02_replay.py tests/test_m6x_d02_boundary_replay.py tests/test_m6x_d02_replay_hang_debug.py
git diff --stat

When done, notify manager pane:
tmux send-keys -t 0:2 'OPUS STEP1_LIVE_NEST_THETA_QV_WIRING DONE - see proofs/v014/step1_live_nest_theta_qv_wiring.md' Enter; sleep 1; tmux send-keys -t 0:2 Enter; sleep 1; tmux send-keys -t 0:2 Enter
