# Tester Report: V0.14 Step-1 Transient Adjust-Base Fix

Date: 2026-06-09

Decision: PASS for the sprint gate. The proof is CPU-only, validates the new
helper against accepted WRF truth, confirms final BaseState semantics remain
unchanged, and the manager reran the requested validation commands.

## Manager Validation

```bash
python -m py_compile src/gpuwrf/integration/d02_replay.py \
  proofs/v014/step1_transient_adjust_base_fix.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/step1_transient_adjust_base_fix.py
python -m json.tool proofs/v014/step1_transient_adjust_base_fix.json \
  >/tmp/step1_transient_adjust_base_fix.manager.validated.json
```

Result: PASS.

Additional narrow Replay smoke:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src pytest -q \
  tests/test_m7_l2_d02_replay.py \
  tests/test_m6x_d02_boundary_replay.py \
  tests/test_m6x_d02_replay_hang_debug.py
```

Result: `4 passed, 2 skipped`.

## Evidence

- `gpu_used=false`
- verdict `STEP1_TRANSIENT_ADJUST_BASE_FIX_THETA_CLOSED`
- source helper additive in `src/gpuwrf/integration/d02_replay.py`
- final BaseState guard unchanged at the known WRF pre-part1 tolerance envelope

## Residual Risk

The helper is not yet wired into production live-nest theta/QV adjustment.
Therefore this sprint closes the transient-adjust-base helper and proof, not
the full Step-1 production initialization path.
