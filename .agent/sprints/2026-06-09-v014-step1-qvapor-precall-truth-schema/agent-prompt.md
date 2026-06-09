You are GPT-5.5 xhigh, schema/debug worker for wrf_gpu2 v0.14.

Read:
- PROJECT_CONSTITUTION.md
- AGENTS.md
- .agent/skills/managing-sprints/SKILL.md
- .agent/sprints/2026-06-09-v014-step1-qvapor-precall-truth-schema/sprint-contract.md

Task:
Establish whether authoritative same-boundary WRF pre-call `QVAPOR` truth
exists for the Step-1 live-nest theta/debug proof. If it exists, produce a
manifest and validator. If it does not, produce the exact minimal WRF savepoint
spec needed next. Do not edit `src/gpuwrf/**`. Do not use GPU. Do not run TOST,
Switzerland, FP32, or memory work. Do not use Hermes.

Important context:
- The accepted pre-call text used by `step1_jax_loader_tstate` appears to
  contain `T/P/PB/MU/MUB/PH/PHB/W` but not `QVAPOR`.
- A candidate `adjust_tempqv` transcription did not close `T_STATE`; do not
  decide on a production fix unless same-boundary truth supports it.
- Existing QVAPOR NPZ artifacts may be post-RK or from a different boundary;
  classify them explicitly instead of reusing them blindly.

Required deliverables:
- `proofs/v014/step1_qvapor_precall_truth_schema.py`
- `proofs/v014/step1_qvapor_precall_truth_schema.json`
- `proofs/v014/step1_qvapor_precall_truth_schema.md`
- `.agent/reviews/2026-06-09-v014-step1-qvapor-precall-truth-schema.md`
- optional `.agent/sprints/2026-06-09-v014-step1-qvapor-precall-truth-schema/artifacts/proposed_wrf_savepoint.md`

Validation:
Run at minimum:
`python -m py_compile proofs/v014/step1_qvapor_precall_truth_schema.py`
`JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_qvapor_precall_truth_schema.py`
`python -m json.tool proofs/v014/step1_qvapor_precall_truth_schema.json >/tmp/step1_qvapor_precall_truth_schema.validated.json`
`git diff -- src/gpuwrf`

Final output:
Write artifacts, then notify manager pane exactly with delayed repeated Enter:
`tmux send-keys -t 0:2 'GPT STEP1_QVAPOR_PRECALL_TRUTH_SCHEMA DONE - see proofs/v014/step1_qvapor_precall_truth_schema.md' Enter; sleep 1; tmux send-keys -t 0:2 Enter; sleep 1; tmux send-keys -t 0:2 Enter`
