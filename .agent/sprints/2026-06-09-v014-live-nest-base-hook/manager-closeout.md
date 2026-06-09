# Manager Closeout

## Outcome

Verdict: `NATIVE_PORT_PLAN_READY`.

The sprint is accepted. It does not fix the grid divergence yet, but it turns
the blocker into an implementation-ready source task. The next production fix
is not a local `PB/MUB` formula tweak and not a CPU-WRF h0 shortcut. It is a
native live-nest initialization stage: WRF parent-to-child interpolation,
`blend_terrain`, then `start_domain_em` base recomputation.

## Proof Objects

- `proofs/v014/live_nest_base_hook.py`
- `proofs/v014/live_nest_base_hook.json`
- `proofs/v014/live_nest_base_hook.md`
- `.agent/reviews/2026-06-09-v014-live-nest-base-hook.md`

## Validation

Manager reran:

```bash
python -m py_compile proofs/v014/live_nest_base_hook.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/live_nest_base_hook.py
python -m json.tool proofs/v014/live_nest_base_hook.json >/tmp/live_nest_base_hook.manager.validated.json
```

Result: pass, `NATIVE_PORT_PLAN_READY`.

## Key Evidence

- Native `wrfinput_d02` vs CPU h0 target-patch deltas:
  - `HGT` max_abs `89.50347900390625` m
  - `PB` max_abs `1047.015625` Pa
  - `MUB` max_abs `1050.3046875` Pa
- WRF base formula on CPU h0 target-patch residuals:
  - `PB` max_abs `0.04889917548280209` Pa
  - `MUB` max_abs `0.044447155625675805` Pa
  - `PHB` max_abs `0.09328280997578986` m2/s2

## Merge Decision

Merge Decision:

Land the proof and roadmap update. The next sprint may modify production source
under a narrowed contract.

## Next Sprint

Open a native source sprint for live-nest base initialization. Requirements:

- no CPU-WRF `wrfout_h0` production dependency;
- no host/device transfers inside timestep loops;
- use CPU-WRF h0 only as validation oracle;
- prove target-patch and whole-domain base-state agreement before resuming TOST.
