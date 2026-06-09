You are GPT-5.5 xhigh, proof/tooling worker for wrf_gpu2 v0.14.

Repository: /home/enric/src/wrf_gpu2
Branch must be worker/gpt/v013-close-manager at current pushed HEAD or newer.
Verify `git log -1` before work.

Read in order:
1. PROJECT_CONSTITUTION.md
2. AGENTS.md
3. .agent/skills/managing-sprints/SKILL.md
4. .agent/sprints/2026-06-09-v014-step1-same-input-truth/sprint-contract.md
5. proofs/v014/same_input_contract_builder.md
6. proofs/v014/same_input_contract_builder.json
7. proofs/v014/same_input_contract_builder.py
8. proofs/v014/wrf_post_rk_refresh_localization_patch.diff
9. proofs/v014/same_state_momentum_mass.py for the existing JAX pre-halo capture call pattern
10. Only source/proof files needed for this task.

Objective:
Produce the full-domain CPU-WRF d02 step-1 post-RK/pre-halo truth npz and run,
or precisely block, the strict same-input WRF-vs-JAX step-1 comparison.

Critical rule:
Do NOT compare WRF step-1 post-RK/pre-halo truth against the JAX initial state.
Accepted comparison is WRF step-1 post-RK/pre-halo vs JAX one-step
`_rk_scan_step_with_pre_halo_capture(...).pre_halo_state` from the same initial
OperationalCarry/Namelist. If that cannot be done, fail closed with the exact
blocker.

Hard rules:
- CPU/JAX proof path first: set `CUDA_VISIBLE_DEVICES=` and `JAX_PLATFORMS=cpu`
  for proof runs.
- No production `src/gpuwrf/**` edits.
- No GPU, no TOST, no Switzerland, no FP32, no memory source work, no
  Hermes/Telegram.
- Patch only a disposable WRF copy under
  /mnt/data/wrf_gpu2/v014_step1_same_input_truth.
- Write only sprint write-scope files plus allowed scratch.
- No weak comparison, no JAX-vs-JAX self-compare, no one-cell proof, and no
  mixed WRF/JAX carry leaves.
- Keep markdown short; put detailed field tables in JSON/CSV if needed.

Required deliverables:
- proofs/v014/step1_same_input_truth.py
- proofs/v014/step1_same_input_truth.json
- proofs/v014/step1_same_input_truth.md
- proofs/v014/step1_same_input_truth_wrf_patch.diff
- optional targeted updates/regeneration of proofs/v014/same_input_contract_builder.py/json/md
- .agent/reviews/2026-06-09-v014-step1-same-input-truth.md

Validation before DONE:
python -m py_compile proofs/v014/step1_same_input_truth.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_same_input_truth.py
python -m json.tool proofs/v014/step1_same_input_truth.json >/tmp/step1_same_input_truth.validated.json
git diff -- src/gpuwrf

If you update same_input_contract_builder.py, also rerun its py_compile, CPU
proof, and JSON validation.

Final review file must include objective, files changed, commands run, proof
objects produced, unresolved risks, and next decision.

When complete, notify manager pane with delayed repeated Enter exactly:
tmux send-keys -t 0:2 'GPT STEP1_SAME_INPUT_TRUTH DONE - see proofs/v014/step1_same_input_truth.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
