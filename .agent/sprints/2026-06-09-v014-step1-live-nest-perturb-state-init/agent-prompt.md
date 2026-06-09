You are GPT-5.5 xhigh, debug worker for wrf_gpu2 v0.14.

Read and obey:

1. `PROJECT_CONSTITUTION.md`
2. `AGENTS.md`
3. `.agent/sprints/2026-06-09-v014-step1-live-nest-perturb-state-init/sprint-contract.md`
4. `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`

Task:

Close the sprint
`.agent/sprints/2026-06-09-v014-step1-live-nest-perturb-state-init`.

Current evidence:

- `proofs/v014/step1_first_rk_part1_p_state_split.json`
- verdict:
  `STEP1_FIRST_RK_PART1_P_STATE_LOCALIZED_PRE_PART1_RAW_CHILD_STATE`
- WRF pre-call to after-part1 is exact for `P/MU/W/PH`.
- JAX `raw_child_state`, `live_child_state`, carry, halo, and
  `_physics_step_forcing.carry.state` all retain
  `P_STATE=69.96875`, `MU_STATE=13.256103515625`,
  `W_STATE=0.7605466246604919` versus WRF pre-call.

Manager hypothesis:

The live-nest child initialization now fixes base/theta/QV but still leaves
`P/MU/W` perturbation state from raw `wrfinput_d02`, while WRF recomputes or
adjusts these before `first_rk_step_part1_call`.

Important: this hypothesis may be wrong. If you disprove it, do not stop at
"blocked"; use your context to rank alternate causes inside the same boundary,
try cheap proof-local falsifiers, and report what you excluded and why.

Constraints:

- No TOST, Switzerland, FP32 source work, memory source work, Hermes, or long
  GPU run.
- Prefer CPU-only.
- Do not edit unrelated files.
- Production source edit allowed only if exact formula/source bug is proven and
  fix is narrow/GPU-native.

Required outputs:

- `proofs/v014/step1_live_nest_perturb_state_init.py`
- `proofs/v014/step1_live_nest_perturb_state_init.json`
- `proofs/v014/step1_live_nest_perturb_state_init.md`
- `.agent/reviews/2026-06-09-v014-step1-live-nest-perturb-state-init.md`

Required validation:

```bash
python -m py_compile proofs/v014/step1_live_nest_perturb_state_init.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/step1_live_nest_perturb_state_init.py
python -m json.tool proofs/v014/step1_live_nest_perturb_state_init.json \
  >/tmp/step1_live_nest_perturb_state_init.validated.json
git diff -- src/gpuwrf
```

At the end, print a concise handoff with objective, files changed, commands run,
proof objects, ranked hypotheses/exclusions, unresolved risks, and next
decision. Then notify manager:

```bash
tmux send-keys -t 0:2 'GPT STEP1_LIVE_NEST_PERTURB_STATE_INIT DONE - see proofs/v014/step1_live_nest_perturb_state_init.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
