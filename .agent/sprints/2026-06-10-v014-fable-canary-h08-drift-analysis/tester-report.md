# Tester Report

Decision: ACCEPT_TESTED.

The manager reran the CPU proof and focused regression gates after Fable wrote
the fix and proof artifacts.

## Commands And Results

- `python -m py_compile src/gpuwrf/integration/d02_replay.py src/gpuwrf/integration/nested_pipeline.py proofs/v014/lbc_cadence_root_cause.py`:
  pass
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src taskset -c 0-3 python proofs/v014/lbc_cadence_root_cause.py`:
  `LBC_CADENCE_ROOT_CAUSE_PROVEN_FIX_GATE_PASS`
- `python -m json.tool proofs/v014/lbc_cadence_root_cause.json`: pass
- `git diff --check`: pass
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src taskset -c 0-7 python -m pytest tests/test_m6_boundary_apply.py tests/test_v013_tost_wrfbdy_fix.py tests/test_p0_1a_nesting.py tests/test_gwd_operational_wiring.py -q`:
  `23 passed, 1 skipped`

## Residual Test Gap

No fixed-commit GPU long gate has run yet. The next required test is a fresh
Canary 72h GPU-vs-CPU field gate with resource CSV logging.
