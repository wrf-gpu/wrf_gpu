# Worker Report

Summary:

Produced a native-port plan for the live-nest base-state split bug.

Objective:

Produce the next oracle or native-port plan for the live-nest base-state split
bug without patching production JAX source.

Outcome:

Verdict: `NATIVE_PORT_PLAN_READY`.

The sprint did not produce a new disposable WRF savepoint. It produced a
source-grounded native implementation plan and repeatable proof script showing:

- native `wrfinput_d02` differs strongly from CPU-WRF h0 over the target patch;
- WRF base formulas on CPU-WRF h0 terrain reproduce `PB/MUB/PHB` within the
  declared formula tolerance;
- the missing production stage is the live-nest path:
  `med_interp_domain` parent interpolation, `blend_terrain`, and
  `start_domain_em` base recomputation.

Files changed:

- `proofs/v014/live_nest_base_hook.py`
- `proofs/v014/live_nest_base_hook.json`
- `proofs/v014/live_nest_base_hook.md`
- `.agent/reviews/2026-06-09-v014-live-nest-base-hook.md`

Commands run:

```bash
python -m py_compile proofs/v014/live_nest_base_hook.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/live_nest_base_hook.py
python -m json.tool proofs/v014/live_nest_base_hook.json >/tmp/live_nest_base_hook.validated.json
```

Proof objects produced:

- `proofs/v014/live_nest_base_hook.json`
- `proofs/v014/live_nest_base_hook.md`
- `.agent/reviews/2026-06-09-v014-live-nest-base-hook.md`

Unresolved risks:

- Native `SINT` parity still needs implementation-level tests.
- `wrfout_h0` lacks `T_INIT/ALB`, so those need formula or future savepoint
  validation.
- The source fix must remain initialization logic and must not introduce
  host/device transfers inside timestep loops.

Next decision:

Dispatch a native source sprint for live-nest base initialization.
