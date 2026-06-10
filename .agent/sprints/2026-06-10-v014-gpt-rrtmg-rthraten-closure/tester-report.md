# Tester Report: V0.14 GPT RRTMG/RTHRATEN Closure

Decision: PASS FOR THE RRTMG FIX; RELEASE GATE STILL RED/BOUNDED.

Commands run by worker and manager:

- `python -m py_compile src/gpuwrf/coupling/physics_couplers.py proofs/v014/rrtmg_rthraten_closure.py proofs/v014/rrtmg_step1_forcing_parity.py proofs/v014/noahmp_step1_closure.py proofs/v014/mynn_rthblten_step1_closure.py`
- `python -m json.tool` on `rrtmg_rthraten_closure.json`, `rrtmg_step1_forcing_parity.json`, `noahmp_step1_closure.json`, and `mynn_rthblten_step1_closure.json`
- `git diff --check` on all touched source/proof/test/review files
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src pytest -q tests/test_v014_dry_source_leaf_wiring.py tests/test_m5_rrtmg_gate.py tests/test_m5_rrtmg_tier1.py tests/test_m5_rrtmg_intermediate_oracles.py tests/test_rrtmg_topographic_coupling.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src taskset -c 4-11 python proofs/v014/rrtmg_rthraten_closure.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src taskset -c 4-11 python proofs/v014/rrtmg_step1_forcing_parity.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src taskset -c 4-11 python proofs/v014/mynn_rthblten_step1_closure.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src taskset -c 4-11 python proofs/v014/noahmp_step1_closure.py`

Results:

- Focused pytest: `14 passed in 12.49s`.
- RRTMG closure verdict: `RRTMG_RTHRATEN_GLW_MOIST_THETA_INPUT_FIXED_REMAINING_RESIDUAL_SPLIT_BOUNDED`.
- RRTMG forcing parity verdict: `RRTMG_STEP1_FORCING_PARITY_MATERIALLY_REDUCED_BY_DRY_THETA_INPUT_FIX`.
- NoahMP strict Step-1 verdict remains red/bounded: `NOAHMP_STEP1_STRICT_RED_FORMALLY_BOUNDED_RRTMG_FIELD_DOMINANT_MYNN_MAX_FLOOR`.

Known issue handled:

- `tests/test_m5_rrtmg_gate.py` was stale and expected RRTMG fallback due SW correctness failure. The current artifact passes Tier-1 SW/LW/Tier-2 and falls back honestly because launch count is `454 > 50`. The test now accepts either correctness-driven fallback or launch-budget fallback.
