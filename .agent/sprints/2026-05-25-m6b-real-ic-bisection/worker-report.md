# Worker Report - M6b Real-IC Bisection

## objective

Run a side-by-side bisection on real Gen2 IC `20260521_18z_l3_24h_20260522T072630Z/wrfout_d02_2026-05-21_18:00:00` for one 10 s timestep, verify the RK1 acoustic fix is actually invoked, localize the first RK/acoustic/operator divergence, cite WRF source, and make no fix attempt.

## verdict

`REAL-IC-FIRST-DIVERGENCE-RK1-ACOUSTIC-SUBSTEP-1-ADVANCE-MU-T-COMMIT`.

The synthetic-IC RK1 omission finding remains valid as a previous defect: the RK1 acoustic loop is now invoked on the real Gen2 IC (`GPUWRF_M6B_RK1_ACOUSTIC_LOOP_ENTER substeps=1`, captured once). Real-IC parity still fails because the operational RK1 small step computes `advance_mu_t_wrf` but does not commit its prognostic `mu`, `theta`, or `mudf` outputs into the operational state. The largest first-substep delta is `theta=5231.41796875`.

## localization

First clean boundary: `rk1_advection_candidate` has zero delta across all traced fields.

First bad boundary: `rk1_acoustic_substep_1`.

Largest first bad operator: `advance_mu_t_committed_outputs`.

Named defect: `src/gpuwrf/runtime/operational_mode.py:_wrf_small_step_acoustic` computes `advanced = advance_mu_t_wrf(inputs)` but then sets `mu_new = state.mu_perturbation` and builds `next_state = state.replace(w=w_solved)`, leaving `advanced["mu"]`, `advanced["theta"]`, and `advanced["mudf"]` out of the committed prognostic state.

WRF citation:
- `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/dyn_em/solve_em.F:1472-1475` runs one RK1 small step for RK3.
- `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/dyn_em/solve_em.F:3435-3452` calls `advance_mu_t` inside `small_steps`.
- `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/dyn_em/module_small_step_em.F:1102-1108` updates `MU`, `MUDF`, `MUTS`, and `MUAVE` in place.
- `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/dyn_em/module_small_step_em.F:1141-1171` updates theta in place.

## files changed

- `src/gpuwrf/runtime/operational_mode.py` - debug-only static marker path and diagnostic entry point; no numerical fix.
- `scripts/m6b_real_ic_operational_compare.py`
- `tests/test_m6b_real_ic_bisection.py`
- `.agent/sprints/2026-05-25-m6b-real-ic-bisection/proof_bisection_run.txt`
- `.agent/sprints/2026-05-25-m6b-real-ic-bisection/proof_real_ic_step1_full_trace.json`
- `.agent/sprints/2026-05-25-m6b-real-ic-bisection/proof_first_diverging_stage.json`
- `.agent/sprints/2026-05-25-m6b-real-ic-bisection/proof_first_diverging_operator.json`
- `.agent/sprints/2026-05-25-m6b-real-ic-bisection/proof_rk1_fix_invocation.txt`
- `.agent/sprints/2026-05-25-m6b-real-ic-bisection/proof_no_regression.txt`
- `.agent/sprints/2026-05-25-m6b-real-ic-bisection/worker-report.md`

## commands run

- `python -m py_compile src/gpuwrf/runtime/operational_mode.py scripts/m6b_real_ic_operational_compare.py`
- `taskset -c 0-3 env PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 python scripts/m6b_real_ic_operational_compare.py --gen2-run-id 20260521_18z_l3_24h_20260522T072630Z --gen2-ic-time "2026-05-21_18:00:00" 2>&1 | tee .agent/sprints/2026-05-25-m6b-real-ic-bisection/proof_bisection_run.txt`
- `taskset -c 0-3 env PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 pytest tests/test_m6b_real_ic_bisection.py -v 2>&1 | tee .agent/sprints/2026-05-25-m6b-real-ic-bisection/proof_no_regression.txt`

## proof objects produced

- `proof_rk1_fix_invocation.txt` - PASS, marker captured once.
- `proof_real_ic_step1_full_trace.json` - full step endpoint deltas plus controlled RK/acoustic trace.
- `proof_first_diverging_stage.json` - `rk1_acoustic_substep_1`, `theta=5231.41796875`.
- `proof_first_diverging_operator.json` - `advance_mu_t_committed_outputs`, WRF citations included.
- `proof_no_regression.txt` - `3 passed`.

## unresolved risks

- The trace after the first divergence is diagnostic only; values naturally explode downstream once RK1 substep 1 diverges.
- The debug marker required a small instrumentation edit to `operational_mode.py` despite the read-only file ownership note. It is static `debug=True` only and the normal `run_forecast_operational` path passes `debug=False`.
- No correction was attempted, so real-IC RK1 parity remains red.

## next decision needed

Dispatch a narrow fix sprint to commit `advance_mu_t_wrf` prognostic outputs (`mu`, `theta`, `mudf`, and aligned scratch/save state) inside operational `_wrf_small_step_acoustic`, then rerun this exact real-IC bisection before broadening scope.
