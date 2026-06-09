# Tester Report

## Tests Added Or Run

- Added: extended `tests/test_operational_namelist_cache_key.py` (5 tests:
  static-holder hashing x2, default-mode aux roundtrip, mixed-mode cache split,
  unknown-mode fail-closed).
- In-proof exactness tests (executed by `mythos_memory_fixes_260609.py`):
  MYNN tile-vs-untiled CPU bit identity (tiles 128/1024, ragged tail, EDMF,
  mixed land/water); moisture velocity reuse function-level exactness
  (active `moist_adv_opt=2`, final RK stage); FP32 R0 contract checks.
- GPU (via `scripts/run_gpu_lowprio.sh`): MYNN tile-vs-untiled bit identity at
  B=40000/tile=4096 on device; compiled-memory measurements; two nested L3 1h
  preflights (baseline lineage + final tree).
- Reran: `proofs/v013/moisture_advection_wiring.py` (5 gates), MYNN pytest
  suite, moisture advection + pd_rk3 operational tests, no-H2D transfer
  audits, acoustic unit tests, flux-advection map factors, pre-halo capture,
  m7 writer + daily pipeline, MYNN-SL oracle parity.

## Results

- Moisture velocity reuse + FP32 R0 exactness proofs: bit-identical
  (max_abs 0.0 across all output leaves).
- MYNN tiling: bit-identical on GPU (B=40000/tile=4096 and the ragged
  production-tile case B=97969/tile=16384). On CPU, non-tridiag fields are
  bit-exact; u/v/theta/qv show <=1.2e-13 (1-2 ulp) scattered-column
  differences from batch-width-dependent XLA:CPU SIMD codegen, reproduced
  with a fresh compilation cache (predeclared codegen bound 5e-13 in the
  proof; NOT a physics tolerance; CPU default paths stay untiled).
- Moisture wiring gates: 5/5 PASS (incl. default byte-identity with the new
  shared-velocity wiring and opt=2 conservation/positivity/stability).
- Pytest battery: 68 passed, 4 skipped (pre-existing skips), 1 failure
  (`test_mynn_fixture_generation_records_harness_binary`) reproduced
  identically on the untouched main worktree — pre-existing environment
  (Fortran harness build) issue, not a regression.
- GPU preflights: `PASS_SHORT_GPU_PREFLIGHT`, payload `PIPELINE_GREEN`,
  all finite, allocator re-exec seen, 0 OOM markers.

## Fixtures Used

- Synthetic production-sized column batches (memory accounting + bit-identity
  only; no physics claims from synthetic inputs).
- Real nested L3 input
  `/mnt/data/canairy_meteo/runs/wrf_l3/20260531_18z_l3_24h_20260601T125256Z`
  for the GPU preflights.
- Closed periodic operational namelist from the v013 wiring proof for the
  moisture gates.

## Gaps

- No long validation / TOST (paused per contract).
- The tiled MYNN path is value-identical by proof; no separate station-skill
  rerun was done (none required for a bit-identical layout change).
- Limiter-workspace and post-physics-merge rows are measured/headroom defers,
  not implementations.

## Decision

Decision: PASS — all required gates green; one pre-existing environment test
failure documented and reproduced on the untouched baseline.
