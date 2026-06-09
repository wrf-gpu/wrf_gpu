You are GPT-5.5 xhigh, proof/debug worker for wrf_gpu2 v0.14.

Repository: /home/enric/src/wrf_gpu2
Branch must be worker/gpt/v013-close-manager at commit d5b541b0 or newer.
Verify `git log -1` before work.

Read in order:
1. PROJECT_CONSTITUTION.md
2. AGENTS.md
3. .agent/skills/managing-sprints/SKILL.md
4. .agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md
5. .agent/sprints/2026-06-09-v014-step1-rk1-source-boundary/sprint-contract.md
6. proofs/v014/step1_t_p_operator_localization.md
7. proofs/v014/step1_t_p_operator_localization.json
8. proofs/v014/step1_t_p_operator_localization.py
9. src/gpuwrf/runtime/operational_mode.py around `_physics_step_forcing`, `_augment_large_step_tendencies`, `_rk_scan_step`
10. Only additional files needed for the task.

Objective:
Split the remaining Step-1 mismatch before `small_step_prep`: WRF
`first_rk_step_part1/part2`, `rk_tendency`, `rk_addtend_dry/spec_bdy_dry`
versus JAX `_physics_step_forcing` and dry tendency construction. Previous proof
localized first material mismatch to RK1 `T_STATE` at
`after_rk_addtend_before_small_step_prep`; RK1 `T_WORK/P_WORK` after
`small_step_prep` matched exactly, so do not debug acoustic yet.

Tooling mandate:
Use the focused substage truth/comparator path. Extend the existing scratch WRF
instrumentation under /mnt/data/wrf_gpu2/v014_step1_rk1_source_boundary if
needed. Do not run long validation campaigns.

Hard rules:
- CPU/JAX proof path first: set `CUDA_VISIBLE_DEVICES=` and `JAX_PLATFORMS=cpu`.
- No GPU, no TOST, no Switzerland, no FP32, no memory source work, no
  Hermes/Telegram.
- No station proxy, no JAX-vs-JAX-only conclusion, no one-cell proof, no
  initial-vs-post-step false comparison.
- Do not modify production `src/gpuwrf/**` unless you prove an exact localized
  bug and a narrow performance-compatible fix inside the sprint contract source
  scope. If you do fix source, rerun the required before/after proofs.
- Keep markdown short; put detailed tables in JSON.

Required deliverables:
- proofs/v014/step1_rk1_source_boundary.py
- proofs/v014/step1_rk1_source_boundary.json
- proofs/v014/step1_rk1_source_boundary.md
- .agent/reviews/2026-06-09-v014-step1-rk1-source-boundary.md
- optional proofs/v014/step1_rk1_source_boundary_wrf_patch.diff

Validation before DONE:
python -m py_compile proofs/v014/step1_rk1_source_boundary.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_rk1_source_boundary.py
python -m json.tool proofs/v014/step1_rk1_source_boundary.json >/tmp/step1_rk1_source_boundary.validated.json
git diff -- src/gpuwrf

If you edit production source, also run the source py_compile command from the
sprint contract and rerun the two previous Step-1 proofs named there.

Final review file must include objective, files changed, commands run, proof
objects produced, unresolved risks, and next decision.

When complete, notify manager pane with delayed repeated Enter exactly:
tmux send-keys -t 0:2 'GPT STEP1_RK1_SOURCE_BOUNDARY DONE - see proofs/v014/step1_rk1_source_boundary.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
