# Worker Report — F1 Real WRF Savepoint Comparator

## objective

Rewrite the M6B6 100-step comparator so it no longer performs a JAX-vs-JAX self-compare, start with WRF Fortran instrumentation assessment, and produce an honest WRF-backed dycore verdict.

## files changed

- `scripts/m6b6_coupled_step_compare.py`
- `tests/savepoint/test_dycore_100_steps.py`
- `tests/savepoint/README.md`
- `tests/savepoint/fixtures/wrf_b6_100step/column/**`
- `proofs/f1/**`
- `.agent/sprints/2026-05-28-f1-wrf-fortran-savepoint-comparator/worker-report.md`

## commands run

- `find /home/enric/src -type f -name wrf.exe -print`
- `find /home/enric/src -type f -name ideal.exe -print`
- `bash external/wrf_savepoint_patch/build.sh`
- `nvidia-smi`
- `python -m py_compile scripts/m6b6_coupled_step_compare.py tests/savepoint/test_dycore_100_steps.py`
- `taskset -c 0-3 python scripts/m6b6_coupled_step_compare.py --tier column --steps 1 --savepoint-root /tmp/f1_smoke --output /tmp/f1_smoke.json`
- `taskset -c 0-3 python scripts/m6b6_coupled_step_compare.py --tier column --steps 100 --savepoint-root /tmp/f1_m6b6_jax --output proofs/f1/m6b6_real_wrf_comparison.json`
- `taskset -c 0-3 pytest -q tests/savepoint/test_dycore_100_steps.py`
- `python .agent/skills/building-wrf-oracles/scripts/validate_fixture_manifest.py tests/savepoint/fixtures/wrf_b6_100step/column/manifest.yaml`

## proof objects produced

- `tests/savepoint/fixtures/wrf_b6_100step/column/manifest.json`
- `tests/savepoint/fixtures/wrf_b6_100step/column/wrf_step001_history_interp_full_timestep.nc` through `wrf_step100_history_interp_full_timestep.nc`
- `proofs/f1/m6b6_real_wrf_comparison.json`
- `proofs/f1/phase1_instrumentation_assessment.md`
- `proofs/f1/dycore_100_steps_pytest.txt`
- `proofs/f1/gpu_status.txt`
- `proofs/f1/honest_dycore_position.md`
- `proofs/f1/fixture_manifest_validation.txt`

## result

`F1_PARTIAL`.

The comparator now reads a separate real-WRF fallback fixture and reports an honest failure instead of a tautological pass. Current HEAD fails at step 1 in `mu` with max abs delta `392.4362662760416 Pa` against tolerance `3e-6 Pa`.

## unresolved risks

- No true Fortran per-RK-stage or per-acoustic-substep savepoints were produced.
- The fallback fixture is derived from hourly WRF history output, so it cannot provide exact stage/substep localization.
- GPU comparison and speed preservation were not proven because the NVIDIA driver is unavailable.
- Baseline-vs-current M11.3 direction was not measured.

## next decision needed

Approve a follow-up sprint to implement real Fortran hook emission and/or restore an `ideal.exe` B6 case so F1 can be upgraded from `F1_PARTIAL` to a real stage-by-stage WRF parity gate.
