You are GPT-5.5 xhigh, debug worker for wrf_gpu2 v0.14.

Read and obey:

1. `PROJECT_CONSTITUTION.md`
2. `AGENTS.md`
3. `.agent/sprints/2026-06-09-v014-step1-start-domain-perturb-subsurface/sprint-contract.md`
4. `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`

Task:

Close the sprint
`.agent/sprints/2026-06-09-v014-step1-start-domain-perturb-subsurface`.

Current evidence:

- `proofs/v014/step1_live_nest_perturb_state_init.json`
- verdict:
  `STEP1_LIVE_NEST_PERTURB_STATE_LOCALIZED_START_DOMAIN_P_PRESS_ADJ_SET_W_SURFACE_P_AL_ALT_SUBSURFACE_GAP`
- Formula reductions:
  - `P_STATE` `69.96875 -> 3.9458582235092763` Pa.
  - `MU_STATE` `13.256103515625 -> 0.047773029698646496` Pa.
  - `W_STATE` `0.7605466246604919 -> 1.2992081932505783e-07` m/s.

Manager hypothesis:

One WRF live-nest `start_domain(nest,.TRUE.)` internal truth surface after the
hypsometric `P/al/alt` recompute and before/after `press_adj` will make the
remaining `P_STATE` and `MU_STATE` contract exact enough to patch JAX safely.

Important: this hypothesis may be wrong. If evidence contradicts it, do not
stop at "blocked"; use your context to rank alternate causes inside the same
boundary, try cheap proof-local falsifiers, and report what you excluded.

Constraints:

- No TOST, Switzerland, FP32 source work, memory source work, Hermes, or GPU run.
- Prefer disposable WRF instrumentation and CPU-only proof replay.
- Do not edit unrelated files.
- Production `src/gpuwrf/integration/d02_replay.py` edit is allowed only if the
  exact formula/source bug is proven and the fix is narrow/GPU-native.

Required outputs:

- `proofs/v014/step1_start_domain_perturb_subsurface.py`
- `proofs/v014/step1_start_domain_perturb_subsurface.json`
- `proofs/v014/step1_start_domain_perturb_subsurface.md`
- `proofs/v014/step1_start_domain_perturb_subsurface_wrf_patch.diff`
- `.agent/reviews/2026-06-09-v014-step1-start-domain-perturb-subsurface.md`

Required validation:

```bash
python -m py_compile proofs/v014/step1_start_domain_perturb_subsurface.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/step1_start_domain_perturb_subsurface.py
python -m json.tool proofs/v014/step1_start_domain_perturb_subsurface.json \
  >/tmp/step1_start_domain_perturb_subsurface.validated.json
git diff -- src/gpuwrf
```

At the end, print a concise handoff with objective, files changed, commands run,
proof objects, ranked hypotheses/exclusions, unresolved risks, and next
decision. Then notify manager:

```bash
tmux send-keys -t 0:2 'GPT STEP1_START_DOMAIN_PERTURB_SUBSURFACE DONE - see proofs/v014/step1_start_domain_perturb_subsurface.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
