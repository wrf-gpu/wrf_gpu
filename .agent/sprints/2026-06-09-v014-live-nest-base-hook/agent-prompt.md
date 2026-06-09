You are GPT-5.5 xhigh acting as a WRF-oracle/debug worker for wrf_gpu2.

Repository: `/home/enric/src/wrf_gpu2`
Branch: `worker/gpt/v013-close-manager`

Read and follow:

1. `PROJECT_CONSTITUTION.md`
2. `AGENTS.md`
3. `.agent/sprints/2026-06-09-v014-live-nest-base-hook/sprint-contract.md`
4. Only the source/proof files needed for this sprint.

Task:

Produce the next oracle or native-port plan for the live-nest base-state split
bug.

Starting facts:

- `proofs/v014/base_state_split_fix.json` verdict is
  `BASE_STATE_SPLIT_FIX_BLOCKED_PARENT_INTERP_BLEND_NOT_LOCAL`.
- The missing production state is WRF's parent-interpolated/blended
  `HGT/MUB/PHB` plus `start_domain_em` base recomputation.
- Do not patch production JAX source in this sprint.
- Do not use CPU-WRF `wrfout_h0` as normal production logic.
- No TOST, Switzerland, FP32, or memory cleanup.
- No Hermes, Telegram, or `ask-hermes`.

Deliver:

- `proofs/v014/live_nest_base_hook.py`
- `proofs/v014/live_nest_base_hook.json`
- `proofs/v014/live_nest_base_hook.md`
- `.agent/reviews/2026-06-09-v014-live-nest-base-hook.md`

Classify as one of:

- `WRF_LIVE_NEST_BASE_ORACLE_READY`
- `NATIVE_PORT_PLAN_READY`
- `LIVE_NEST_BASE_HOOK_BLOCKED_<reason>`

Required validation:

```bash
python -m py_compile proofs/v014/live_nest_base_hook.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/live_nest_base_hook.py
python -m json.tool proofs/v014/live_nest_base_hook.json \
  >/tmp/live_nest_base_hook.validated.json
```

When done, print:

`GPT LIVE_NEST_BASE_HOOK DONE`
