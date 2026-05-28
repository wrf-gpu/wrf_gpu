# M11.1 Worker Report

Verdict: `M11P1_COMPLETE`

## Objective

Triage whether `p_perturbation` and `ph_perturbation` should be active under `dycore_rk3`, apply the minimal fix, and re-run the required 3-step harness plus 100-step parity guard.

## Decision

Decision class: `PROGNOSTIC_BUG_FIX_DYCORE`.

WRF treats `ph_perturbation` as nonhydrostatic acoustic state advanced by `advance_w`; WRF treats `p_perturbation` as diagnostic pressure, but it is refreshed inside the acoustic dycore after `ph/theta/mu` changes. The GPU harness expectation that both resident leaves change during `dycore_rk3` is valid.

## Files Changed

- `src/gpuwrf/dynamics/core/acoustic.py`
- `src/gpuwrf/contracts/state.py`
- `.agent/sprints/2026-05-28-m11p1-dycore-pressure-geopotential-triage/triage.md`
- `.agent/sprints/2026-05-28-m11p1-dycore-pressure-geopotential-triage/worker-report.md`
- `proofs/m11p1/diagnostic_report_after_fix.json`
- `proofs/m11p1/dycore_100_steps_pytest.json`

## Commands Run

- `python -m py_compile src/gpuwrf/dynamics/core/acoustic.py src/gpuwrf/contracts/state.py` -> pass
- `taskset -c 0-3 env XLA_PYTHON_CLIENT_MEM_FRACTION=0.10 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=true PYTHONPATH=src python scripts/run_diagnostic_harness.py --hours 0.008333333333333333 --jax-platform gpu --output proofs/m11p1/diagnostic_report_after_fix.json` -> failed before run; this JAX install requires `cuda`, not `gpu`, for `JAX_PLATFORMS`
- `taskset -c 0-3 env XLA_PYTHON_CLIENT_MEM_FRACTION=0.10 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=true PYTHONPATH=src python scripts/run_diagnostic_harness.py --hours 0.008333333333333333 --jax-platform cuda --output proofs/m11p1/diagnostic_report_after_fix.json` -> pass
- `taskset -c 0-3 pytest -q tests/savepoint/test_dycore_100_steps.py` -> pass, `1 passed in 451.53s (0:07:31)`

## Proof Objects Produced

- `proofs/m11p1/diagnostic_report_after_fix.json`: `dycore_rk3` verdict is `ACTIVE`; `p_perturbation` and `ph_perturbation` are no longer `NOISY_ZERO`.
- `proofs/m11p1/dycore_100_steps_pytest.json`: records the required 100-step pytest pass.

## Post-Fix Harness Headline

`NOISY_ZERO operators (partial coupling failure): microphysics_thompson [6/7 expected fields have delta = 0 across the run: qs, qv, qc, qg, qi, qr]`

This is outside M11.1 writable scope and is in M17 territory. Relative to the smoking-gun proof, `dycore_rk3` moved from `NOISY_ZERO` to `ACTIVE`.

No non-dycore operator regressed in the 3-step smoke: `surface_layer`, `mynn_pbl`, and `lateral_boundary` remain `ACTIVE`; `rrtmg` remains cadence-`INACTIVE`; guards remain `PASSIVE_OK`; Thompson remains the only `NOISY_ZERO` operator.

## Unresolved Risks

- The `p_perturbation` refresh is a reduced dycore diagnostic refresh, not a full WRF `calc_p_rho` equation-of-state implementation with explicit `alt/c2a` intermediates.
- The harness proof was generated from the working tree before the final commit, so the embedded `commit` field names the pre-commit HEAD.

## Next Decision Needed

Decide whether a follow-up sprint should port the full WRF `calc_p_rho` diagnostic into the resident acoustic core once the needed pressure/intermediate interface is explicitly frozen.
