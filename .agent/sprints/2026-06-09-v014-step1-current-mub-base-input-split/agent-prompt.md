You are GPT-5.5 xhigh, WRF/JAX proof debugger for wrf_gpu2 v0.14.

Read:
- PROJECT_CONSTITUTION.md
- AGENTS.md
- .agent/skills/managing-sprints/SKILL.md
- .agent/sprints/2026-06-09-v014-step1-current-mub-base-input-split/sprint-contract.md
- proofs/v014/step1_adjust_tempqv_intermediate.md
- proofs/v014/step1_adjust_tempqv_intermediate.json
- proofs/v014/step1_theta_same_qvapor.md
- proofs/v014/step1_jax_loader_tstate.md
- .agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md

Task: explain the current-MUB/base-input mismatch that drives the remaining
Step-1 live-nest theta residual. The target is d02 Fortran {i:18,j:10,k:2},
zero {k:1,y:9,x:17}. Previous proof: saved inputs match, but current `mub`
differs by about 17.675 Pa and `pb_new`/`p_new` by about 17.494 Pa.

Do not edit src/gpuwrf/**. Do not use GPU. Do not run TOST, Switzerland, FP32
source work, memory source work, or Hermes.

Allowed scratch WRF tree:
/mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/WRF

Allowed scratch root:
/mnt/data/wrf_gpu2/v014_step1_current_mub_base_input_split

Required deliverables:
- proofs/v014/step1_current_mub_base_input_split.py
- proofs/v014/step1_current_mub_base_input_split.json
- proofs/v014/step1_current_mub_base_input_split.md
- proofs/v014/step1_current_mub_base_input_split_wrf_patch.diff
- .agent/reviews/2026-06-09-v014-step1-current-mub-base-input-split.md

Proof goal:
- Freeze and verify the formula `p_new = p + c3h * (mub + p_top) + c4h`.
- Inspect WRF live-nest path around `mub_save`, `blend_terrain`, and
  `adjust_tempqv`.
- Emit/recover WRF current `mub`, `mub_save`, `pb_new`/equivalent, terrain/base
  inputs, and pre/post-`blend_terrain` values for the target cell or a compact
  neighborhood.
- Recompute the same values on the JAX/proof side using the actual
  live-nest base-init path used by the theta proof.
- Classify with exactly one sprint verdict from the contract and recommend the
  next smallest justified source-changing sprint, or a precise blocker.

Validation:
python -m py_compile proofs/v014/step1_current_mub_base_input_split.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_current_mub_base_input_split.py
python -m json.tool proofs/v014/step1_current_mub_base_input_split.json >/tmp/step1_current_mub_base_input_split.validated.json
git diff -- src/gpuwrf

When done, notify manager pane exactly with delayed repeated Enter:
tmux send-keys -t 0:2 'GPT STEP1_CURRENT_MUB_BASE_INPUT_SPLIT DONE - see proofs/v014/step1_current_mub_base_input_split.md' Enter; sleep 1; tmux send-keys -t 0:2 Enter; sleep 1; tmux send-keys -t 0:2 Enter
