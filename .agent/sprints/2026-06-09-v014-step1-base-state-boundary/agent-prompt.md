You are GPT-5.5 xhigh, debug worker for wrf_gpu2 v0.14.

Read and obey:

1. `PROJECT_CONSTITUTION.md`
2. `AGENTS.md`
3. `.agent/skills/managing-sprints/SKILL.md`
4. `.agent/sprints/2026-06-09-v014-step1-base-state-boundary/sprint-contract.md`
5. `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`

Task:

Close sprint `.agent/sprints/2026-06-09-v014-step1-base-state-boundary`.

Current evidence:

- `proofs/v014/step1_jax_start_domain_input_split.json`
- Verdict:
  `STEP1_JAX_START_DOMAIN_INPUT_SPLIT_LOCALIZED_BASE_STATE_RECONSTRUCTION_FP32_ALT_SOURCE_ORDER_GAP`.
- Direct WRF ALT substitution reduces P max_abs
  `3.9458582235092763 -> 0.07605321895971429`.
- FP32 ALT diagnosis with WRF `PHB+MUB` reduces pressure max_abs to `0.0859375`.
- Best local WRF-order fp32/cp=1004.5 base candidate still leaves
  `P_STATE` max_abs `2.828125` and `MU_STATE` max_abs `0.011962890625`.

Manager hypothesis:

The remaining gap is exact WRF base-state reconstruction/source order before
the hypsometric `AL/ALT` pass in `start_domain_em`, especially `PHB+MUB`.

Important:

If this hypothesis is wrong, do not stop at "blocked"; use your context to rank
alternate causes, try cheap proof-local falsifiers, and report what you
excluded.

Constraints:

- No TOST, Switzerland, FP32 production source work, memory production source
  work, Hermes, or GPU run.
- Prefer CPU-only proof replay and disposable WRF instrumentation under
  `/mnt/data/wrf_gpu2/`.
- Do not edit production `src/gpuwrf/**`.
- If you use WRF instrumentation, write the patch diff to
  `proofs/v014/step1_base_state_boundary_wrf_patch.diff`.

Required outputs:

- `proofs/v014/step1_base_state_boundary.py`
- `proofs/v014/step1_base_state_boundary.json`
- `proofs/v014/step1_base_state_boundary.md`
- `.agent/reviews/2026-06-09-v014-step1-base-state-boundary.md`
- optional `proofs/v014/step1_base_state_boundary_wrf_patch.diff`

Required validation:

```bash
python -m py_compile proofs/v014/step1_base_state_boundary.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/step1_base_state_boundary.py
python -m json.tool proofs/v014/step1_base_state_boundary.json \
  >/tmp/step1_base_state_boundary.validated.json
git diff -- src/gpuwrf
```

At the end, print a concise handoff with objective, files changed, commands
run, proof objects, ranked hypotheses/exclusions, unresolved risks, and next
decision. Then notify manager:

```bash
tmux send-keys -t 0:2 'GPT STEP1_BASE_STATE_BOUNDARY DONE - see proofs/v014/step1_base_state_boundary.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
