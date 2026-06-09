You are GPT-5.5 xhigh, proof/debug worker for wrf_gpu2 v0.14.

Repository: /home/enric/src/wrf_gpu2
Branch must be worker/gpt/v013-close-manager at commit 980700e2 or newer.
Verify `git log -1` before work.

Read in order:
1. PROJECT_CONSTITUTION.md
2. AGENTS.md
3. .agent/skills/managing-sprints/SKILL.md
4. .agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md
5. .agent/sprints/2026-06-09-v014-step1-t-p-operator-localization/sprint-contract.md
6. proofs/v014/step1_live_nest_init_rerun.md
7. proofs/v014/step1_live_nest_init_rerun.json
8. proofs/v014/step1_live_nest_init_rerun.py
9. src/gpuwrf/runtime/operational_mode.py around `_physics_step_forcing`, `_rk_scan_step`, `_augment_large_step_tendencies`, `_acoustic_scan`, `_refresh_grid_p_from_finished`
10. Only additional files needed for the task.

Objective:
Localize the remaining Step-1 strict same-input divergence after live-nest base
init closure. Known result: base fields are closed; first divergent field is T;
largest residual is P. Find the earliest dynamic/operator boundary or deliver a
narrow source fix with before/after proof.

Tooling mandate:
Use the fastest rigorous wall-clock path. Prefer a focused Step-1 substage
truth/comparator harness over slow runtime reproduction. If WRF substage truth
is needed, instrument only a disposable WRF copy under
/mnt/data/wrf_gpu2/v014_step1_t_p_operator_localization.

Hard rules:
- CPU/JAX proof path first: set `CUDA_VISIBLE_DEVICES=` and `JAX_PLATFORMS=cpu`.
- No GPU, no TOST, no Switzerland, no FP32, no memory source work, no
  Hermes/Telegram.
- No station proxy, no JAX-vs-JAX-only conclusion, no one-cell proof, no
  initial-vs-post-step false comparison.
- Do not modify production `src/gpuwrf/**` unless you have an exact localized
  bug and a narrow performance-compatible fix inside the sprint contract's
  source scope. If you do fix source, rerun the strict Step-1 proof after.
- Keep markdown short; put detailed tables in JSON.

Required deliverables:
- proofs/v014/step1_t_p_operator_localization.py
- proofs/v014/step1_t_p_operator_localization.json
- proofs/v014/step1_t_p_operator_localization.md
- .agent/reviews/2026-06-09-v014-step1-t-p-operator-localization.md
- optional proofs/v014/step1_t_p_operator_localization_wrf_patch.diff

Validation before DONE:
python -m py_compile proofs/v014/step1_t_p_operator_localization.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_t_p_operator_localization.py
python -m json.tool proofs/v014/step1_t_p_operator_localization.json >/tmp/step1_t_p_operator_localization.validated.json
git diff -- src/gpuwrf

If you edit production source, also run the source py_compile command from the
sprint contract and rerun:
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_live_nest_init_rerun.py
python -m json.tool proofs/v014/step1_live_nest_init_rerun.json >/tmp/step1_live_nest_init_rerun.after_tp_fix.validated.json

Final review file must include objective, files changed, commands run, proof
objects produced, unresolved risks, and next decision.

When complete, notify manager pane with delayed repeated Enter exactly:
tmux send-keys -t 0:2 'GPT STEP1_TP_OPERATOR_LOCALIZATION DONE - see proofs/v014/step1_t_p_operator_localization.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
