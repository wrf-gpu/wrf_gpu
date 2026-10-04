# v0.20 S1/S2 Test Fixture Migration Report

Date: 2026-06-21
Branch: `worker/integration/v020`
Worker: codex

## Objective

Migrate the named test fixtures from removed legacy `State` leaves / kwargs
`p`, `ph`, and `mu` to the authoritative `p_total`, `ph_total`, and
`mu_total` leaves. Production code was not modified.

## Files changed

- `tests/test_m6x_c2_scan.py`
  - Replaced removed `_state_field_shapes()` keys `p`/`ph`/`mu` with
    `p_total`/`ph_total`/`mu_total`.
  - Kept the test's analytic perturbation intent by assigning
    `p_perturbation = p_total`, `ph_perturbation = ph_total`, and
    `mu_perturbation = mu_total`.
- `tests/test_noahmp_checkpoint_v2.py`
  - Built `Tendencies(p/ph/mu)` buffers from the total leaf shapes:
    `p_total`, `ph_total`, `mu_total`.
- `tests/test_v013_mrf_operational.py`
- `tests/test_v013_myj_janjic_operational.py`
- `tests/test_v013_sfclay_pbl_pairing.py`
- `tests/test_v013_t3_surface_lsm_wiring.py`
- `tests/test_v017_gfs_pbl_operational.py`
  - Replaced legacy constructor kwargs `p=`, `ph=`, `mu=` with
    `p_total=`, `ph_total=`, `mu_total=` for the physical pressure,
    geopotential, and dry-column-mass fixtures.

## Before / after failure count

- Before: integration brief diagnosed 14 new failures on top of 53
  pre-existing failures, caused by stale test fixtures after S1 removed the
  legacy `p`/`ph`/`mu` leaves.
- After: full CPU sweep returns to exactly the 53 pre-existing failures:
  `53 failed, 1748 passed, 375 skipped, 35 xfailed, 2 xpassed`.
- No migrated fixture appears in the final failure list.
- No `KeyError: 'p'`, `KeyError: 'ph'`, `KeyError: 'mu'`, or migrated-fixture
  NaN signature appears in `proofs/v020/s1s2/pytest_full_cpu.log`.

## Commands run

All Python / pytest commands were pinned to cores 0-3 with
`OMP_NUM_THREADS=4`; pytest runs also used `JAX_PLATFORMS=cpu`.

```bash
taskset -c 0-3 env OMP_NUM_THREADS=4 JAX_PLATFORMS=cpu python -m pytest -q \
  tests/test_m6x_c2_scan.py \
  tests/test_noahmp_checkpoint_v2.py \
  tests/test_v013_mrf_operational.py \
  tests/test_v013_myj_janjic_operational.py \
  tests/test_v013_sfclay_pbl_pairing.py \
  tests/test_v013_t3_surface_lsm_wiring.py \
  tests/test_v017_gfs_pbl_operational.py
```

Result: `74 passed, 1 skipped, 2 xpassed in 28.60s`.

```bash
taskset -c 0-3 env OMP_NUM_THREADS=4 JAX_PLATFORMS=cpu python -m pytest -q tests/test_m5_rrtmg_tier1.py
taskset -c 0-3 env OMP_NUM_THREADS=4 JAX_PLATFORMS=cpu python -m pytest -q tests/test_v0130_compile_speed.py
```

Results: `2 passed in 10.12s`; `22 passed in 21.15s`.

```bash
taskset -c 0-3 env OMP_NUM_THREADS=4 JAX_PLATFORMS=cpu python -m pytest -q tests --tb=short
```

Result: expected red suite only:
`53 failed, 1748 passed, 375 skipped, 35 xfailed, 2 xpassed, 5 warnings in 1830.81s`.
Raw log: `proofs/v020/s1s2/pytest_full_cpu.log` (ignored by `.gitignore`).

```bash
taskset -c 0-3 env OMP_NUM_THREADS=4 python proofs/v020/s1s2/harness.py compare --steps 6
```

Result:

```text
[compare] 71/71 leaves BITWISE-EQUAL (6 steps)
[compare] leaves only in baseline (removed): ['state.mu', 'state.p', 'state.ph']
[compare] RESULT: BIT-IDENTICAL on all retained leaves
```

Note: the existing S1/S2 baseline was captured at 6 steps (see
`gate_a_bit_identical.txt`). The unqualified harness default is 4 steps, which
does not match that baseline; no baseline artifact was overwritten.

## Failure classification

The remaining 53 failures are the expected pre-existing red bucket: missing AIFS
Vtable / WRF radiation source tables / v0.17-v0.18 oracle artifacts, the
honest-red manifest/status checks, `test_agentos_smoke`,
`test_m7_1km_memory_audit`, and the paper author-disclosure tripwire.

No failure found during this migration is a real dycore bug.

## Proof objects

- `proofs/v020/s1s2/TEST_MIGRATION_REPORT.md`
- `proofs/v020/s1s2/TESTFIX_DONE`
- `proofs/v020/s1s2/pytest_full_cpu.log` (ignored raw command log)
- `proofs/v020/s1s2/harness_compare_steps6.log` (ignored raw command log)
