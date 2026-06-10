# Tester Report: V0.14 Fable NoahMP Step-1 Closure

Decision: PASS FOR NARROWING; STRICT STEP-1 STILL RED.

Manager-rerun acceptance gates:

```bash
python -m py_compile proofs/v014/step1_mynn_source_coupling.py proofs/v014/step1_surface_land_flux_handoff.py proofs/v014/noahmp_step1_closure.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src taskset -c 0-7 python proofs/v014/noahmp_step1_closure.py
python -m json.tool proofs/v014/noahmp_step1_closure.json >/tmp/noahmp_step1_closure.validated.json
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src taskset -c 0-7 python proofs/v014/step1_mynn_source_coupling.py
python -m json.tool proofs/v014/step1_mynn_source_coupling.json >/tmp/step1_mynn_source_coupling.validated.json
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src taskset -c 0-7 python proofs/v014/step1_surface_land_flux_handoff.py
python -m json.tool proofs/v014/step1_surface_land_flux_handoff.json >/tmp/step1_surface_land_flux_handoff.validated.json
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src taskset -c 0-7 pytest -q tests/test_v014_mynn_surface_layer_regressions.py tests/test_m6_surface_layer_kernel.py tests/test_v014_dry_source_leaf_wiring.py tests/test_v014_mynn_coldstart_init.py tests/test_noahmp_coupler.py
git diff --check
```

Result:

- Proofs executed and JSON validated.
- `pytest`: 16 passed, 1 pre-existing skip.
- `git diff --check`: clean.
- No GPU, TOST, Switzerland, or long validation run was started.

Important caveat:

- The strict release gate is still red. This sprint is accepted as a precise
  WRF-anchored narrowing plus a small production fix, not as grid-parity
  closure.
