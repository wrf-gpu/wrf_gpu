You are Opus 4.8 xhigh, correctness-critical source worker for wrf_gpu2 v0.14.

Read:
- PROJECT_CONSTITUTION.md
- AGENTS.md
- .agent/skills/managing-sprints/SKILL.md
- .agent/sprints/2026-06-09-v014-step1-transient-adjust-base-fix/sprint-contract.md
- proofs/v014/step1_current_mub_base_input_split.md
- proofs/v014/step1_current_mub_base_input_split.json
- proofs/v014/step1_theta_same_qvapor.md
- proofs/v014/step1_theta_same_qvapor.json
- src/gpuwrf/integration/d02_replay.py

Task: implement the smallest source fix for the Step-1 live-nest theta/QV
initialization mismatch. WRF `adjust_tempqv` uses transient post-`blend_terrain`
/ pre-`start_domain` current `MUB`; final post-`start_domain` BaseState must
stay unchanged. The previous proof shows direct WRF blend MUB matches the WRF
adjust hook, while final BaseState MUB matches pre-part1 truth.

Allowed production source:
- src/gpuwrf/integration/d02_replay.py

Required proof artifacts:
- proofs/v014/step1_transient_adjust_base_fix.py
- proofs/v014/step1_transient_adjust_base_fix.json
- proofs/v014/step1_transient_adjust_base_fix.md
- .agent/reviews/2026-06-09-v014-step1-transient-adjust-base-fix.md

Do not edit unrelated production files. Do not use GPU. Do not run TOST,
Switzerland, FP32 source work, memory source work, or Hermes.

Proof requirements:
- Transient adjust-base `MUB` matches WRF adjust hook at target cell.
- Final BaseState `MUB` remains matched to WRF pre-part1 final target.
- Corrected theta/QV candidate is compared against same-boundary WRF pre-call
  truth.
- Verdict is exactly one from the sprint contract.

Validation:
python -m py_compile src/gpuwrf/integration/d02_replay.py proofs/v014/step1_transient_adjust_base_fix.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_transient_adjust_base_fix.py
python -m json.tool proofs/v014/step1_transient_adjust_base_fix.json >/tmp/step1_transient_adjust_base_fix.validated.json
git diff --stat

When done, try to notify manager pane with delayed repeated Enter:
tmux send-keys -t 0:2 'OPUS STEP1_TRANSIENT_ADJUST_BASE_FIX DONE - see proofs/v014/step1_transient_adjust_base_fix.md' Enter; sleep 1; tmux send-keys -t 0:2 Enter; sleep 1; tmux send-keys -t 0:2 Enter
If tmux notification fails, record it in your final handoff and leave artifacts on disk.
