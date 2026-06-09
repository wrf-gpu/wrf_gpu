# Tester Report

Decision: pass.

Verdict: `PASS`.

Validation commands:

```bash
python -m py_compile proofs/v014/live_nest_base_hook.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/live_nest_base_hook.py
python -m json.tool proofs/v014/live_nest_base_hook.json >/tmp/live_nest_base_hook.manager.validated.json
```

Result:

- Proof script compiles.
- Proof script runs CPU-only and emits `NATIVE_PORT_PLAN_READY`.
- JSON validates.
- No production JAX source was edited.
- No GPU, TOST, Switzerland, FP32, or long validation run was launched.

Key numeric checks:

- Native `wrfinput_d02` vs CPU h0 target patch:
  - `HGT` max_abs `89.50347900390625` m
  - `PB` max_abs `1047.015625` Pa
  - `MUB` max_abs `1050.3046875` Pa
  - `PHB` max_abs `878.0291748046875` m2/s2
- WRF base formula on CPU h0 target patch:
  - `PB` max_abs residual `0.04889917548280209` Pa
  - `MUB` max_abs residual `0.044447155625675805` Pa
  - `PHB` max_abs residual `0.09328280997578986` m2/s2

Residual risk:

This is a source-port plan, not a production fix. The next sprint must prove the
native interpolation/blend/recompute implementation against these gates.
