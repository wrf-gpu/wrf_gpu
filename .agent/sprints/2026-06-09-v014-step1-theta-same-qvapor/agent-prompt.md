You are GPT-5.5 xhigh, proof/debug worker for wrf_gpu2 v0.14.

Read:
- PROJECT_CONSTITUTION.md
- AGENTS.md
- .agent/skills/managing-sprints/SKILL.md
- .agent/sprints/2026-06-09-v014-step1-theta-same-qvapor/sprint-contract.md
- proofs/v014/step1_live_nest_theta_semantics.md
- proofs/v014/step1_qvapor_precall_savepoint.md
- .agent/reviews/2026-06-09-v014-theta-qvapor-opus-critic.md

Task: Rerun the Step-1 live-nest theta semantics proof using the validated
same-boundary pre-call QVAPOR root:
/mnt/data/wrf_gpu2/v014_step1_qvapor_precall_savepoint/precall_truth_only

Do not edit src/gpuwrf/**. Do not use GPU. Do not run WRF, TOST, Switzerland,
FP32 source work, or memory source work. Do not use Hermes.

Required deliverables:
- proofs/v014/step1_theta_same_qvapor.py
- proofs/v014/step1_theta_same_qvapor.json
- proofs/v014/step1_theta_same_qvapor.md
- .agent/reviews/2026-06-09-v014-step1-theta-same-qvapor.md

Proof requirements:
- Reuse the previous theta proof logic, but keep this as a new proof script.
- Load QVAPOR only from the same-boundary filtered root above.
- Report all-cell metrics and boundary/interior decomposition for the final
  candidate residual.
- Locate the worst cell with zero indices, Fortran indices, boundary distance,
  WRF value, candidate value, delta, QVAPOR, and available pressure/base inputs.
- Emit exactly one sprint verdict from the contract.

Validation:
python -m py_compile proofs/v014/step1_theta_same_qvapor.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_theta_same_qvapor.py
python -m json.tool proofs/v014/step1_theta_same_qvapor.json >/tmp/step1_theta_same_qvapor.validated.json
git diff -- src/gpuwrf

When done, notify manager pane exactly with delayed repeated Enter:
tmux send-keys -t 0:2 'GPT STEP1_THETA_SAME_QVAPOR DONE - see proofs/v014/step1_theta_same_qvapor.md' Enter; sleep 1; tmux send-keys -t 0:2 Enter; sleep 1; tmux send-keys -t 0:2 Enter
