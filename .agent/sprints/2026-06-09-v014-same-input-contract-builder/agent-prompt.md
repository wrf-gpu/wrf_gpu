You are GPT-5.5 xhigh, proof/tooling worker for wrf_gpu2 v0.14.

Repository: /home/enric/src/wrf_gpu2
Branch must be worker/gpt/v013-close-manager at current pushed HEAD or newer.
Verify `git log -1` before work.

Read in order:
1. PROJECT_CONSTITUTION.md
2. AGENTS.md
3. .agent/skills/managing-sprints/SKILL.md
4. .agent/sprints/2026-06-09-v014-same-input-contract-builder/sprint-contract.md
5. proofs/v014/early_step_discriminator.md
6. proofs/v014/early_step_discriminator.json
7. Only source/proof files needed for this task.

Objective:
Build the missing same-input comparison contract/tooling, then rerun a strict
early-step candidate comparison if technically possible. This is a tooling
sprint, not a production source-fix sprint.

Hard rules:
- CPU/JAX proof path first: set `CUDA_VISIBLE_DEVICES=` and `JAX_PLATFORMS=cpu`
  for proof runs.
- No production `src/gpuwrf/**` edits.
- No GPU, no TOST, no Switzerland, no FP32, no memory source work, no
  Hermes/Telegram.
- Write only the sprint write-scope files plus scratch under
  /mnt/data/wrf_gpu2/v014_same_input_contract_builder.
- No weak comparison, no JAX-vs-JAX self-compare, no one-cell proof, and no
  mixing JAX-produced carry with WRF leaves.
- Keep markdown short; detailed tables go in JSON/CSV if needed.

Required deliverables:
- proofs/v014/same_input_contract_builder.py
- proofs/v014/same_input_contract_builder.json
- proofs/v014/same_input_contract_builder.md
- optional proofs/v014/same_input_contract_builder_wrf_patch.diff
- optional updates/regeneration of proofs/v014/early_step_discriminator.py/json/md
- .agent/reviews/2026-06-09-v014-same-input-contract-builder.md

Validation before DONE:
python -m py_compile proofs/v014/same_input_contract_builder.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/same_input_contract_builder.py
python -m json.tool proofs/v014/same_input_contract_builder.json >/tmp/same_input_contract_builder.validated.json
git diff -- src/gpuwrf

If you update early_step_discriminator.py, also rerun its py_compile, CPU proof,
and JSON validation.

Final review file must include objective, files changed, commands run, proof
objects produced, unresolved risks, and next decision.

When complete, notify manager pane with delayed repeated Enter exactly:
tmux send-keys -t 0:2 'GPT SAME_INPUT_CONTRACT_BUILDER DONE - see proofs/v014/same_input_contract_builder.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
