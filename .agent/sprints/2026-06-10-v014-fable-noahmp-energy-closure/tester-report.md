# Tester Report

Decision: PASS FOR THIS SPRINT, STRICT RELEASE GATE STILL RED.

Manager reran the sprint gates after the Fable handoff:

```bash
python -m py_compile proofs/v014/noahmp_step1_closure.py proofs/v014/noahmp_land_tile_energy_closure.py
python -m json.tool proofs/v014/noahmp_step1_closure.json >/tmp/noahmp_step1_closure.manager.validated.json
python -m json.tool proofs/v014/noahmp_land_tile_energy_closure.json >/tmp/noahmp_land_tile_energy_closure.manager.validated.json
git diff --check -- src/gpuwrf/physics/noahmp_coupler.py tests/test_noahmp_coupler.py proofs/v014/noahmp_step1_closure.py proofs/v014/noahmp_step1_closure.md proofs/v014/noahmp_step1_closure.json proofs/v014/noahmp_land_tile_energy_closure.py proofs/v014/noahmp_land_tile_energy_closure.md proofs/v014/noahmp_land_tile_energy_closure.json .agent/reviews/2026-06-10-v014-fable-noahmp-energy-closure.md
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src taskset -c 4-7 python proofs/v014/noahmp_land_tile_energy_closure.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src taskset -c 4-7 python proofs/v014/noahmp_step1_closure.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src taskset -c 4-7 pytest -q tests/test_noahmp_coupler.py tests/test_v014_mynn_surface_layer_regressions.py tests/test_m6_surface_layer_kernel.py tests/test_v014_dry_source_leaf_wiring.py tests/test_v014_mynn_coldstart_init.py
```

Results:

- `noahmp_land_tile_energy_closure.py`: reproduced
  `NOAHMP_LAND_TILE_ENERGY_CLOSED_NARROWED_TO_RRTMG_RADIATION_FORCING`.
- `noahmp_step1_closure.py`: reproduced
  `NOAHMP_STEP1_WIRED_STRICT_RED_NARROWED_TO_RADIATION_FORCING_INTO_NOAHMP`.
- Pytest subset: `17 passed, 1 skipped in 28.42s`.

The sprint-specific gate passes. The v0.14 release gate remains blocked until
surface-layer/MYNN water-path theta semantics and RRTMG forcing parity are
closed or explicitly bounded.
