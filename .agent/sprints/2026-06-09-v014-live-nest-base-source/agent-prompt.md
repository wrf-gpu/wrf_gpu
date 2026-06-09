You are GPT-5.5 xhigh acting as a source-fix worker for wrf_gpu2.

Repository: `/home/enric/src/wrf_gpu2`
Branch: `worker/gpt/v013-close-manager`

Read and follow:

1. `PROJECT_CONSTITUTION.md`
2. `AGENTS.md`
3. `.agent/sprints/2026-06-09-v014-live-nest-base-source/sprint-contract.md`
4. Only the source/proof files needed for this sprint.

Task:

Implement or precisely block the native source fix for the live-nested d02
base-state initialization mismatch.

Starting facts:

- `proofs/v014/live_nest_base_hook.json` verdict is `NATIVE_PORT_PLAN_READY`.
- Native `wrfinput_d02` vs CPU-WRF h0 has target-patch deltas about
  `PB=1047` Pa and `MUB=1050` Pa.
- WRF base formulas on CPU-WRF h0 terrain reproduce `PB/MUB/PHB` within `0.1`
  native units.
- The required WRF chain is `med_interp_domain` ->
  `nest_interpdown_interp.inc` / `interp_fcn_sint` / `sint.F` ->
  `blend_terrain` -> `start_domain_em`.
- Existing `src/gpuwrf/nesting/interp.py` already has WRF `sint`
  registration and full-child interpolation helpers; inspect before adding new
  interpolation machinery.

Hard rules:

- Do not use CPU-WRF `wrfout_h0` as production input; validation oracle only.
- Do not add host/device transfers inside timestep loops.
- Do not run TOST, Switzerland validation, FP32, or memory cleanup.
- Keep the GPU-native high-performance concept intact. Initialization work is
  acceptable before timestepping; timestep-loop corrections are not.
- No Hermes, Telegram, or `ask-hermes`.
- Keep terminal output compact; detailed tables go in JSON.

Deliver:

- source patch if practical and narrow;
- `proofs/v014/live_nest_base_source_fix.py`
- `proofs/v014/live_nest_base_source_fix.json`
- `proofs/v014/live_nest_base_source_fix.md`
- `.agent/reviews/2026-06-09-v014-live-nest-base-source-fix.md`

Classify as:

- `LIVE_NEST_BASE_SOURCE_FIXED`
- `LIVE_NEST_BASE_SOURCE_PARTIAL_<reason>`
- `LIVE_NEST_BASE_SOURCE_BLOCKED_<reason>`

Required validation:

```bash
python -m py_compile \
  src/gpuwrf/integration/d02_replay.py \
  src/gpuwrf/integration/nested_pipeline.py \
  src/gpuwrf/nesting/interp.py \
  src/gpuwrf/nesting/boundary_construction.py \
  proofs/v014/live_nest_base_source_fix.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/live_nest_base_source_fix.py
python -m json.tool proofs/v014/live_nest_base_source_fix.json \
  >/tmp/live_nest_base_source_fix.validated.json
```

When done, print:

`GPT LIVE_NEST_BASE_SOURCE DONE`

Then notify the manager tmux window without using Hermes:

```bash
tmux send-keys -t 0:2 'GPT LIVE_NEST_BASE_SOURCE DONE - see proofs/v014/live_nest_base_source_fix.md' Enter
```
