You are GPT-5.5 xhigh, debug worker for wrf_gpu2 v0.14.

Read and obey:

1. `PROJECT_CONSTITUTION.md`
2. `AGENTS.md`
3. `.agent/sprints/2026-06-09-v014-step1-first-rk-part1-p-state-split/sprint-contract.md`
4. `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`

Task:

Close the sprint
`.agent/sprints/2026-06-09-v014-step1-first-rk-part1-p-state-split`.

The predecessor proof is closed at commit `ebedb3c1`:
`STEP1_P_PH_MU_LOCALIZED_FIRST_RK_STEP_PART1_P_STATE`.
It found WRF `after_first_rk_step_part1` vs JAX
`_physics_step_forcing.carry.state`, `P_STATE` max_abs `69.96875`, with
`MU_STATE` and `W_STATE` material at the same checked boundary. RK1
`small_step_prep` / `calc_p_rho(step=0)` work arrays are exact.

Your objective is to split inside WRF `first_rk_step_part1`, especially around
`phy_prep` and `calc_p_rho_phi`, or to name the exact missing surface/contract.
Prefer one decisive CPU-only savepoint/comparator over slow free-running runs.

Required outputs:

- `proofs/v014/step1_first_rk_part1_p_state_split.py`
- `proofs/v014/step1_first_rk_part1_p_state_split.json`
- `proofs/v014/step1_first_rk_part1_p_state_split.md`
- `.agent/reviews/2026-06-09-v014-step1-first-rk-part1-p-state-split.md`

Optional:

- `proofs/v014/step1_first_rk_part1_p_state_split_wrf_patch.diff`
- `proofs/v014/step1_first_rk_part1_p_state_split_source_patch.diff`

Constraints:

- No TOST, Switzerland, FP32 source work, memory source work, Hermes, or long
  GPU forecast.
- Prefer CPU-only. Do not use GPU unless a manager explicitly authorizes it.
- Do not edit unrelated files.
- Do not make production source edits unless the exact bug is proven and the
  fix is narrow and GPU-performance-compatible.

Required validation:

```bash
python -m py_compile proofs/v014/step1_first_rk_part1_p_state_split.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/step1_first_rk_part1_p_state_split.py
python -m json.tool proofs/v014/step1_first_rk_part1_p_state_split.json \
  >/tmp/step1_first_rk_part1_p_state_split.validated.json
git diff -- src/gpuwrf
```

At the end, print a concise handoff with objective, files changed, commands
run, proof objects, unresolved risks, and next decision. Then notify manager:

```bash
tmux send-keys -t 0:2 'GPT STEP1_FIRST_RK_PART1_P_STATE_SPLIT DONE - see proofs/v014/step1_first_rk_part1_p_state_split.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
