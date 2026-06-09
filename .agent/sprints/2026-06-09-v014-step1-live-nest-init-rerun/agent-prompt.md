You are GPT-5.5 xhigh, proof/falsifier worker for wrf_gpu2 v0.14.

Repository: /home/enric/src/wrf_gpu2
Branch must be worker/gpt/v013-close-manager at current pushed HEAD or newer.
Verify `git log -1` before work.

Read in order:
1. PROJECT_CONSTITUTION.md
2. AGENTS.md
3. .agent/skills/managing-sprints/SKILL.md
4. .agent/sprints/2026-06-09-v014-step1-live-nest-init-rerun/sprint-contract.md
5. proofs/v014/step1_same_input_truth.md
6. proofs/v014/step1_same_input_truth.json
7. proofs/v014/step1_same_input_truth.py
8. proofs/v014/live_nest_base_source_fix.md
9. proofs/v014/live_nest_base_source_fix.json
10. src/gpuwrf/integration/d02_replay.py and nested_pipeline.py only as needed.

Objective:
Rerun the strict d02 step-1 same-input comparison with the existing production
native live-nest child base initialization path wired into the proof loader.

Critical method:
Do not compare WRF step-1 truth to JAX initial state. Do not headline raw
wrfinput d02. Accepted comparison is CPU-WRF step-1 post-RK/pre-halo truth
versus JAX one-step `_rk_scan_step_with_pre_halo_capture(...).pre_halo_state`
from a live-nest-initialized d02 carry.

Hard rules:
- CPU-only proof: `JAX_PLATFORMS=cpu`, `CUDA_VISIBLE_DEVICES=`.
- No GPU, no TOST, no Switzerland, no FP32, no memory source work, no Hermes.
- Reuse the existing truth npz. Do not rebuild WRF unless you prove it is invalid.
- Prefer proof-local loader wiring. Edit production source only if you prove a
  narrow defect in the existing live-nest init path.
- No weak comparison, no JAX-vs-JAX self-compare, no one-cell proof.
- Keep markdown short; put detailed tables in JSON.

Required deliverables:
- proofs/v014/step1_live_nest_init_rerun.py
- proofs/v014/step1_live_nest_init_rerun.json
- proofs/v014/step1_live_nest_init_rerun.md
- .agent/reviews/2026-06-09-v014-step1-live-nest-init-rerun.md

Validation before DONE:
python -m py_compile proofs/v014/step1_live_nest_init_rerun.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_live_nest_init_rerun.py
python -m json.tool proofs/v014/step1_live_nest_init_rerun.json >/tmp/step1_live_nest_init_rerun.validated.json
git diff -- src/gpuwrf

If you edit production source, also run the source py_compile set and
`proofs/v014/live_nest_base_source_fix.py` as specified in the sprint contract.

Final review file must include objective, files changed, commands run, proof
objects produced, unresolved risks, and next decision.

When complete, notify manager pane with delayed repeated Enter exactly:
tmux send-keys -t 0:2 'GPT STEP1_LIVE_NEST_INIT_RERUN DONE - see proofs/v014/step1_live_nest_init_rerun.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
