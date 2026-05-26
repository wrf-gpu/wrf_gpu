# Worker Report — M6 Guard-Disabled Debug Impl

Summary: Implemented the diagnostic `disable_guards` lane in operational mode and produced the guard-disabled proof objects. With guards disabled on the pinned 20260521 IC, the first >10x-envelope excursion is `p_perturbation` at step 49, cell `[32, 52, 40]`, value `-314176.084627652` Pa, ratio `62.8352169255304` versus the 5 kPa envelope. The first traced operator boundary that produces the excursion is `horizontal_pressure_gradient`.

## Hypothesis vs actual

Hypothesis from tester/critic: hard operational guards, especially `theta = physical_origin.theta`, were hiding the real prognostic failure.

Actual: disabling the projection/masking guards lets theta, moisture, mass, and pressure evolve through the diagnostic path. The first 10x failure is pressure, not theta: `p_perturbation` crosses the 10x envelope at step 49. The diagnostic trace localizes the first bad boundary to the RK tendency update labelled `horizontal_pressure_gradient`.

## Files changed

- `src/gpuwrf/runtime/operational_mode.py`
- `scripts/m6_guard_disabled_debug.py`
- `.agent/sprints/2026-05-26-m6-guard-disabled-debug-impl/worker-report.md`
- `.agent/sprints/2026-05-26-m6-guard-disabled-debug-impl/proof_*.json`
- `.agent/sprints/2026-05-26-m6-guard-disabled-debug-impl/validation_*.txt`
- `.agent/sprints/2026-05-26-m6-guard-disabled-debug-impl/v3_521_default/*`
- Compatibility copies of the four `proof_*.json` files under `.agent/sprints/2026-05-26-m6-guard-disabled-debug/`, because the committed acceptance scaffold reads that tester sprint path.

## Commands run and output

`python -m py_compile scripts/m6_guard_disabled_debug.py src/gpuwrf/runtime/operational_mode.py`

Output: no stdout/stderr, exit 0.

`PYTHONPATH=src python - <<'PY' ... OperationalNamelist defaults ... PY`

Output: `disable_guards False`, exit 0.

`OMP_NUM_THREADS=4 PYTHONPATH=src taskset -c 0-3 python scripts/m6_guard_disabled_debug.py --run-id 20260521_18z_l3_24h_20260522T072630Z --n-steps 75 --output .agent/sprints/2026-05-26-m6-guard-disabled-debug-impl/`

Output captured in `validation_m6_guard_disabled_debug.txt`: status `OK`; first explosive step `p_perturbation`, step `49`, cell `[32, 52, 40]`, value `-314176.084627652`; first operator `horizontal_pressure_gradient`.

`OMP_NUM_THREADS=4 PYTHONPATH=src taskset -c 0-3 pytest tests/test_m6_guard_disabled_debug.py -v`

Output captured in `validation_pytest_m6_guard_disabled_debug.txt`: `12 passed in 0.93s`.

`OMP_NUM_THREADS=4 PYTHONPATH=src taskset -c 0-3 python scripts/m6b6_coupled_step_compare.py --tier all`

Output captured in `validation_m6b6_coupled_step_compare.txt`: `passed True`; outcome `SEVENTH-COUPLED-STEP-PARITY-ACHIEVED`; global max_abs_delta `0.0`; failures `0`; transfer audit inside timestep loop `0` bytes for column/golden/patch16.

`OMP_NUM_THREADS=4 PYTHONPATH=src taskset -c 0-3 python scripts/m6b_v3_localize_521.py --run-id 20260521_18z_l3_24h_20260522T072630Z --output .agent/sprints/2026-05-26-m6-guard-disabled-debug-impl/v3_521_default/`

Output captured in `validation_m6b_v3_localize_521.txt`: status `OK`; verdict `NAMED-FIX:boundary_application`; generated `proof_step46_violation.json` reports first bad step `46`, `V_max = 11.480101585388184` m/s at cell `[2, 1, 0]`.

## Proof objects

- `.agent/sprints/2026-05-26-m6-guard-disabled-debug-impl/proof_guard_inventory.json`
- `.agent/sprints/2026-05-26-m6-guard-disabled-debug-impl/proof_guards_off_safe_default.json`
- `.agent/sprints/2026-05-26-m6-guard-disabled-debug-impl/proof_first_explosive_step.json`
- `.agent/sprints/2026-05-26-m6-guard-disabled-debug-impl/proof_first_explosive_operator.json`
- `.agent/sprints/2026-05-26-m6-guard-disabled-debug-impl/v3_521_default/proof_step46_violation.json`
- `.agent/sprints/2026-05-26-m6-guard-disabled-debug-impl/validation_m6b6_coupled_step_compare.txt`
- `.agent/sprints/2026-05-26-m6-guard-disabled-debug-impl/validation_pytest_m6_guard_disabled_debug.txt`

## Risks

- The operator trace is a diagnostic decomposition of the current operational wrapper, not WRF Fortran operator parity.
- The acceptance test path mismatch required mirrored proof JSONs in the prior tester sprint directory; no existing tester report or contract file was modified.
- The named operator is an available RK boundary label. It narrows the next sprint to the pressure/RK tendency path, but the exact WRF source term still needs savepoint-level comparison.

## Handoff

Objective complete. Next decision: dispatch a pressure/RK tendency localization or WRF savepoint comparison sprint focused on the `horizontal_pressure_gradient` / pressure-perturbation update around 20260521 step 49, cell `[32, 52, 40]`.
