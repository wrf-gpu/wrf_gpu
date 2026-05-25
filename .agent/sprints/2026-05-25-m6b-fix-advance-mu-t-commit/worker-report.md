# Worker Report - M6b Fix advance_mu_t Commit

## objective

Fix the four contracted operational `_wrf_small_step_acoustic` defects in one pass: promote `advance_mu_t_wrf` outputs, recompute W coefficients per substep, use acoustic-cadence `dt_sub`, and match the validation-bound `ph_tend` formula. Verify the real-IC parity and bounded probes on 4 CPU cores with sanitizer off.

## verdict

`FAILED-KILL-GATE-NEW-FIFTH-DEFECT`.

The four contracted code edits were applied, but the required gates do not pass. Step-1 controlled parity remains red after tiny scratch-order differences amplify through the theta recurrence, and the direct 10 s operational bounded probe goes nonfinite. I did not commit because the sprint contract requires all gates to pass before closeout.

## per-defect diff summary

1. Prognostic/scratch promotion: `_wrf_small_step_acoustic` now commits `advanced["mu"]` into `State.mu/mu_total/mu_perturbation`, commits `advanced["theta"]` through `State.theta`, and carries `mudf/muts/muave/ww/t_2ave/ph_tend` in resident `OperationalCarry`. WRF citations: `solve_em.F:3435-3452`, `module_small_step_em.F:1102-1108`, `module_small_step_em.F:1141-1171`.
2. W coefficient recompute: the operational substep recomputes `calc_coef_w_wrf_coefficients(...)` inside `_wrf_small_step_acoustic` from resident `carry.muts`. WRF citation: `solve_em.F:2409-2717`.
3. `dt_sub`: acoustic substep duration now uses `dt_s / acoustic_substeps`; RK2 cadence was also aligned with WRF half sound steps. WRF citation: `solve_em.F:1472-1479`.
4. `ph_tend`: operational carry now uses the validation-bound theta-delta increment and keeps `ph_tend` FP64. WRF consumption citation: `module_small_step_em.F:1345-1395`.

## proof results

- `proof_step1_parity.txt` / `proof_step1_parity_after_fix.json`: FAIL. Final max delta `5.62949953421312e14`; theta-family fields fail, other listed fields are 0.0.
- `proof_step10_probe.txt`: FAIL. `scripts/m6b_carry_expansion_probe.py --runs 1 --duration-s 10` reports `NONFINITE`.
- `tests/test_m6b_fix_advance_mu_t_commit.py`: PASS, 4 tests.
- `python -m py_compile src/gpuwrf/runtime/operational_state.py src/gpuwrf/runtime/operational_mode.py scripts/m6b_real_ic_operational_compare.py`: PASS.

## likely fifth defect

Once theta is promoted, the existing operational call feeds raw `state.u/state.v` into `advance_mu_t_wrf`. The WRF call site passes `grid%ru_m`, `grid%rv_m`, and `grid%ww_m` into `advance_mu_t` (`solve_em.F:3439-3441`), not raw velocity components. The previous carry-only implementation masked this because the explosive theta output was never committed to prognostic state.

## files changed

- `src/gpuwrf/runtime/operational_mode.py`
- `src/gpuwrf/runtime/operational_state.py`
- `scripts/m6b_real_ic_operational_compare.py`
- `tests/test_m6b_fix_advance_mu_t_commit.py`
- `.agent/sprints/2026-05-25-m6b-fix-advance-mu-t-commit/proof_step1_parity.txt`
- `.agent/sprints/2026-05-25-m6b-fix-advance-mu-t-commit/proof_step1_parity_after_fix.json`
- `.agent/sprints/2026-05-25-m6b-fix-advance-mu-t-commit/proof_step10_probe.txt`
- `.agent/sprints/2026-05-25-m6b-fix-advance-mu-t-commit/worker-report.md`

## commands run

- `python -m py_compile src/gpuwrf/runtime/operational_state.py src/gpuwrf/runtime/operational_mode.py scripts/m6b_real_ic_operational_compare.py`
- `taskset -c 0-3 env PYTHONPATH=src OMP_NUM_THREADS=4 pytest tests/test_m6b_fix_advance_mu_t_commit.py -v`
- `taskset -c 0-3 env PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 python scripts/m6b_real_ic_operational_compare.py --gen2-run-id 20260521_18z_l3_24h_20260522T072630Z --steps 1`
- `taskset -c 0-3 env PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 python scripts/m6b_carry_expansion_probe.py --runs 1 --duration-s 10`

## unresolved risks

- The four requested fixes expose a deeper unit/composition defect in the operational acoustic theta path.
- B6 and full no-regression were not run after the kill gate failed.
- The 70 s probe was not run because the 10 s bounded probe already went nonfinite.

## next decision needed

Dispatch a follow-up sprint to replace raw velocity inputs to `advance_mu_t_wrf` with WRF-shaped `ru_m/rv_m/ww_m` equivalents, or explicitly decide that operational theta promotion must wait until the full small-step momentum composition is ported.
