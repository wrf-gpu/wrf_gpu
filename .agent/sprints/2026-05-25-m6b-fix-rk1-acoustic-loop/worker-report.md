# Worker Report - M6b Fix RK1 Acoustic Loop

## objective

Add the missing operational RK1 acoustic small-step loop in `src/gpuwrf/runtime/operational_mode.py`, preserve the validation-only module boundary, and verify against the sprint gates.

## verdict

`PARTIAL-FIX-COMMITTED-BUT-M6B-RETRY-V3-NOT-READY`.

The operational RK1 dispatch now runs one acoustic small step, matching WRF RK3 stage-1 cadence in `solve_em.F:1472-1475` (`dt_rk = dt/3`, `number_of_small_timesteps = 1`). Validation mode stayed clean: B6 golden still reports `SEVENTH-COUPLED-STEP-PARITY-ACHIEVED` with `max_abs_delta: 0.0`.

The acceptance gates are not fully closed. The requested 1-step and 10-step operational-vs-validation comparator for run `20260523_18z_l3_24h_20260524T004313Z` is blocked locally because that run directory has `wrfinput_d02` and `wrfbdy_d01` but fewer than two `wrfout_d02*` history files. The 70-second probe is finite but fails theta/wind bounds.

## implementation summary

- Added a `substeps` override to operational `_acoustic_scan`.
- Changed RK1 from `advance_stage(..., use_acoustic=False)` to `advance_stage(..., acoustic_substeps=1)`.
- Kept RK2/RK3 on the existing `namelist.acoustic_substeps` operational cadence.
- Added `tests/test_m6b_fix_rk1_acoustic_loop.py` to lock RK1 acoustic dispatch and prevent validation-only imports from `operational_mode.py`.
- Did not import `gpuwrf.dynamics.acoustic_loop` or `gpuwrf.dynamics.coupled_step`.

## proof summary

| gate | artifact | status |
|---|---|---|
| Step 1 parity | `proof_step1_parity.json`, `proof_step1_parity.txt` | BLOCKED: missing `wrfout_d02*` history files for requested run |
| Step 10 parity | `proof_step10_parity.json`, `proof_step10_parity.txt` | BLOCKED: same missing requested-run history |
| 70-second probe | `proof_step70_probe.json`, `proof_step70_probe.txt` | FAIL: finite, but `theta_bounded=false`, `wind_bounded=false` |
| B6 regression | `proof_b6_regression.txt` | PASS: `SEVENTH-COUPLED-STEP-PARITY-ACHIEVED`, repeated `max_abs_delta: 0.0` |
| No regression | `proof_no_regression.txt` | FAIL: 152 passed, 1 unrelated missing-artifact failure in `test_warmed_capture_artifacts_present` |

## files changed

- `src/gpuwrf/runtime/operational_mode.py`
- `tests/test_m6b_fix_rk1_acoustic_loop.py`
- `.agent/sprints/2026-05-25-m6b-fix-rk1-acoustic-loop/proof_step1_parity.json`
- `.agent/sprints/2026-05-25-m6b-fix-rk1-acoustic-loop/proof_step1_parity.txt`
- `.agent/sprints/2026-05-25-m6b-fix-rk1-acoustic-loop/proof_step10_parity.json`
- `.agent/sprints/2026-05-25-m6b-fix-rk1-acoustic-loop/proof_step10_parity.txt`
- `.agent/sprints/2026-05-25-m6b-fix-rk1-acoustic-loop/proof_step70_probe.json`
- `.agent/sprints/2026-05-25-m6b-fix-rk1-acoustic-loop/proof_step70_probe.txt`
- `.agent/sprints/2026-05-25-m6b-fix-rk1-acoustic-loop/proof_b6_regression.txt`
- `.agent/sprints/2026-05-25-m6b-fix-rk1-acoustic-loop/proof_no_regression.txt`
- `.agent/sprints/2026-05-25-m6b-fix-rk1-acoustic-loop/worker-report.md`

## commands run

- `python -m py_compile src/gpuwrf/runtime/operational_mode.py tests/test_m6b_fix_rk1_acoustic_loop.py`
- `taskset -c 0-3 env PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 pytest tests/test_m6b_fix_rk1_acoustic_loop.py -v`
- `taskset -c 0-3 env PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 python scripts/m6b6_coupled_step_compare.py --tier golden`
- `taskset -c 0-3 env PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 pytest tests/test_m6x_*.py tests/test_m3_transfer_audit.py tests/test_m6b0_*.py tests/test_m6b0r_*.py tests/test_m6b1_*.py tests/test_m6b2_*.py tests/test_m6b3_*.py tests/test_m6b_hygiene_*.py tests/test_m6b4_*.py tests/test_m6b5_*.py tests/test_m6b6_*.py tests/test_m6_operational_*.py tests/test_m6_perf_*.py tests/test_m6b_carry_expansion_*.py tests/test_m6b_d2h_warmed_*.py tests/test_m6b_honest_v2_*.py tests/test_m6b_operational_vs_validation_*.py tests/test_m6b_fix_rk1_*.py -v`
- Redirected `scripts.m6b_operational_vs_validation_compare.run_bisection(...)` to this sprint directory for 1-step and 10-step proof attempts.
- Redirected `scripts.m6b_carry_expansion_probe.run_probe(..., duration_s=70.0)` to this sprint directory for the 70-second probe.

## proof objects produced

- `proof_step1_parity.json`
- `proof_step1_parity.txt`
- `proof_step10_parity.json`
- `proof_step10_parity.txt`
- `proof_step70_probe.json`
- `proof_step70_probe.txt`
- `proof_b6_regression.txt`
- `proof_no_regression.txt`

## unresolved risks

- The requested parity fixture run is incomplete in local storage, so the 1-step `<1e-10` and 10-step `<1e-8` gates were not measurable.
- Running RK1 acoustic scratch without promoting `advance_mu_t_wrf` theta/mu into operational prognostic state may not be enough to close the original theta divergence. A direct theta/mu promotion attempt was tested locally and backed out because it made the 70-second probe nonfinite; that needs a separate contracted design/fix.
- The full regression command still depends on an unrelated D2H warmed Nsight artifact: `.agent/sprints/2026-05-25-m6b-d2h-warmed-recapture/proof_warmed.nsys-rep`.

## next decision needed

Do not dispatch M6b RETRY V3 from this proof set. Restore or regenerate the missing `wrfout_d02*` history files for `20260523_18z_l3_24h_20260524T004313Z`, then rerun the 1-step/10-step comparator. If divergence remains, dispatch a narrowly scoped operational acoustic prognostic-state sprint with explicit approval to update theta/mu/p/ph consistency rather than only RK1 cadence.
