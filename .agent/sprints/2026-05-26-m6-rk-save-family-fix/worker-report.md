Summary: Implemented the M6 RK save-family mass-basis fix. The matched defect was a MUTS/MU accumulation basis error: acoustic substeps were carrying physical perturbation MU where WRF advances a small-step delta, so `c1h*muts+c2h` could collapse near zero and blow up theta projection. No denominator clamp was added.

Files changed:
- `src/gpuwrf/dynamics/mu_t_advance.py`
- `src/gpuwrf/dynamics/core/acoustic.py`
- `src/gpuwrf/runtime/operational_mode.py`
- `tests/test_m6_rk_save_family_fix.py`
- `.agent/sprints/2026-05-26-m6-rk-save-family-fix/proof_*.json`
- `.agent/sprints/2026-05-26-m6-rk-save-family-fix/validation_*.txt`
- `.agent/sprints/2026-05-26-m6-rk-save-family-fix/worker-report.md`

Fix summary:
- `advance_mu_t_wrf` now derives the acoustic working MU from `muts-mut`, advances that delta, forms `muts=mut+mu_work_new`, and computes `muave` on the same basis.
- `acoustic_substep_core` exposes `mu` as `advanced["muts"] - state.mut`, matching the small-step delta contract.
- `_with_save_family` resets RK/acoustic scratch `muts` to `mu_base` and `muave` to zero at stage prep; `_carry_from_acoustic_core` reconstructs physical `mu_total` from fixed base plus `muts-mut`.
- The previous WRF theta mass-coupling algebra was left intact.

Commands run and output:
- `taskset -c 0-3 python scripts/m6_guard_disabled_debug.py --run-id 20260521_18z_l3_24h_20260522T072630Z --n-steps 50 --output .agent/sprints/2026-05-26-m6-rk-save-family-fix/baseline/`
  Output: baseline reproduced theta failure at step 47, cell `[11,31,67]`, value `-226584368.0`, first operator `acoustic`.
- `taskset -c 0-3 python scripts/m6_guard_disabled_debug.py --run-id 20260521_18z_l3_24h_20260522T072630Z --n-steps 360 --output .agent/sprints/2026-05-26-m6-rk-save-family-fix/fixed/`
  Output captured in `validation_guard_disabled_360.txt`: `status: OK`; driver hard-capped at step 75; worst theta ratio `0.8097948346819196`.
- Uncapped 120-step internal guard-disabled probe.
  Output captured in `validation_delta_mass_no_cap_120.txt`: no 10x envelope breach through 120 steps; worst theta step 120 ratio `0.9304928152901786`.
- `taskset -c 0-3 python scripts/m6b6_coupled_step_compare.py --tier all`
  Output captured in `validation_m6b6_coupled_step_compare.txt`: `passed: true`, `outcome: SEVENTH-COUPLED-STEP-PARITY-ACHIEVED`, `diverging_field_count: 0`.
- `taskset -c 0-3 python scripts/m6b_real_ic_operational_compare.py --steps 10`
  Output captured in `validation_m6b_real_ic_operational_compare_steps10.txt`: `status: PASS`, `final_max_abs_delta: 0.0`.
- `taskset -c 0-3 pytest tests/test_m6_guard_disabled_debug.py -v`
  Output captured in `validation_pytest_guard_disabled_debug.txt`: `12 passed`.
- `taskset -c 0-3 pytest tests/test_m6_rk_save_family_fix.py -v`
  Output captured in `validation_pytest_rk_save_family_fix.txt`: `1 passed`.

Proof objects produced:
- `.agent/sprints/2026-05-26-m6-rk-save-family-fix/proof_baseline_reproduces.json`
- `.agent/sprints/2026-05-26-m6-rk-save-family-fix/proof_muts_trajectory.json`
- `.agent/sprints/2026-05-26-m6-rk-save-family-fix/proof_wrf_fortran_crosscheck.json`
- `.agent/sprints/2026-05-26-m6-rk-save-family-fix/proof_no_cap_120.json`
- `.agent/sprints/2026-05-26-m6-rk-save-family-fix/proof_fix_validation.json`
- `.agent/sprints/2026-05-26-m6-rk-save-family-fix/baseline/proof_first_explosive_step.json`
- `.agent/sprints/2026-05-26-m6-rk-save-family-fix/fixed/proof_first_explosive_step.json`

Risks:
- The public guard-disabled driver accepts `--n-steps 360` but internally runs `min(n_steps, 75)`. I produced an additional uncapped 120-step proof to satisfy the sprint fallback, but not a full 360-step uncapped forecast.
- The DMDT sign conversion is based on the local Python C-grid orientation with WRF negative `dnw`; it is protected by B6 and 10-step real-IC parity, but should remain visible to review.

Handoff:
- Objective met: denominator collapse is fixed without a clamp, B6 and real-IC parity are preserved, and the regression test catches the step-46 denominator at the historical bad cell.
- Next decision needed: reviewer should decide whether the guard-disabled diagnostic driver’s 75-step cap should be removed in a separate sprint so future 360-step acceptance commands mean what they request.
- Remote push was not performed because the sprint contract lists `NO remote push` as a non-goal; branch work is ready locally for manager integration.
