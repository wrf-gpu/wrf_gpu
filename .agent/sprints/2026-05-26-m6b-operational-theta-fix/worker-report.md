# Worker Report

Summary: BLOCKED_ACCEPTANCE_NOT_MET. I fixed the controlled operational theta
nonfinite in the shared acoustic path, but the sprint's full Stage 2 acceptance
is not met because the 20260509 1h Canary replay still fails at step 11 with
`MATH:coftz`.

## Summary

The matched defect was in the operational/shared acoustic theta transition:
`advance_mu_t_wrf` produced WRF's mass-coupled small-step theta value, and
`acoustic_substep_core` carried it forward as perturbation theta. I added the
WRF-style decoupling formula at the acoustic composition boundary and switched
the theta flux source in `advance_mu_t_wrf` to the running theta-average source.
This makes real-IC CPU parity steps 2/5/10 pass with final max delta 0.0 and all
fields finite. B6 coupled-step parity also remains green with
`SEVENTH-COUPLED-STEP-PARITY-ACHIEVED`.

The unresolved blocker is outside the fixed parity symptom: 20260509 still
breaches theta at step 11 with the existing `MATH:coftz` classification.

## Files Changed

- `src/gpuwrf/dynamics/mu_t_advance.py`
- `src/gpuwrf/dynamics/core/acoustic.py`
- `tests/test_m6b_operational_theta_fix.py`
- `.agent/sprints/2026-05-26-m6b-operational-theta-fix/hypothesis_notes.md`
- `.agent/sprints/2026-05-26-m6b-operational-theta-fix/proof_baseline_reproduces.json`
- `.agent/sprints/2026-05-26-m6b-operational-theta-fix/proof_fix_validation.json`
- `.agent/sprints/2026-05-26-m6b-operational-theta-fix/*_step*.txt`
- `.agent/sprints/2026-05-26-m6b-operational-theta-fix/v3_521*`
- `.agent/sprints/2026-05-26-m6b-operational-theta-fix/v3_509*`
- `.agent/sprints/2026-05-26-m6b-operational-theta-fix/pytest_*.txt`

## Commands Run + Output

- `taskset -c 0-3 python scripts/m6b_real_ic_operational_compare.py --steps 2`
  - Baseline output: `status: FAIL`; step 2 `theta`, `t_2ave`, and `ph_tend` had `max_abs_delta: 1e+300`, `all_fields_finite: false`.
- `taskset -c 0-3 python scripts/m6b_real_ic_operational_compare.py --steps 2`
  - Fixed output: `status: PASS`; final `max_abs_delta: 0.0`; step 2 all fields finite.
- `taskset -c 0-3 python scripts/m6b_real_ic_operational_compare.py --steps 5`
  - Output: `status: PASS`; final `max_abs_delta: 0.0`; all fields finite.
- `taskset -c 0-3 python scripts/m6b_real_ic_operational_compare.py --steps 10`
  - Output: `status: PASS`; final `max_abs_delta: 0.0`; all fields finite.
- `taskset -c 0-3 python scripts/m6b6_coupled_step_compare.py --tier all`
  - Output: `passed: true`; `outcome: SEVENTH-COUPLED-STEP-PARITY-ACHIEVED`; field deltas 0.0.
- `taskset -c 0-3 python scripts/m6b_v3_localize_521.py --run-id 20260521_18z_l3_24h_20260522T072630Z --output .../v3_521/`
  - Output: `status: OK`; `verdict: NAMED-FIX:boundary_application`.
- `taskset -c 0-3 python scripts/m6b_v3_localize_509.py --run-id 20260509_18z_l3_24h_20260511T190519Z --output .../v3_509/`
  - Output: `status: MATH:coftz`; first violation step 11, theta around `2.44e12 K` at k=28 j=59 i=72.
- `taskset -c 0-3 pytest tests/test_m6b_operational_theta_fix.py -v`
  - Output: `1 passed in 56.42s`.
- `taskset -c 0-3 pytest -x`
  - Output: stopped at the pre-existing missing external fixture failure:
    `tests/test_canary_wrf_fixture.py::test_full_external_file_exists_at_external_uri`,
    missing `data/fixtures/canary-wrf-d01-20260518T18-tslice-v1/full.npz`.

## Proof Objects

- `.agent/sprints/2026-05-26-m6b-operational-theta-fix/proof_baseline_reproduces.json`
- `.agent/sprints/2026-05-26-m6b-operational-theta-fix/proof_fix_validation.json`
- `.agent/sprints/2026-05-26-m6b-operational-theta-fix/fixed_step2.txt`
- `.agent/sprints/2026-05-26-m6b-operational-theta-fix/fixed_step5.txt`
- `.agent/sprints/2026-05-26-m6b-operational-theta-fix/fixed_step10.txt`
- `.agent/sprints/2026-05-26-m6b-operational-theta-fix/b6_coupled_step_compare.txt`
- `.agent/sprints/2026-05-26-m6b-operational-theta-fix/v3_521/`
- `.agent/sprints/2026-05-26-m6b-operational-theta-fix/v3_509/`
- `.agent/sprints/2026-05-26-m6b-operational-theta-fix/hypothesis_notes.md`

## Risks

- Full acceptance is blocked: the 20260509 path still fails the 1h theta bound.
- The remaining 20260509 signature includes explosive microphysics species
  (`qc`) and is still classified by the localizer as `MATH:coftz`; this patch
  should not be represented as an operational 1h fix.
- `pytest -x` cannot run to completion in this worktree because the external
  fixture file is absent.

## Handoff

Objective: fix operational theta path drift in `advance_mu_t_wrf` /
`acoustic_substep_core`.

Files changed: listed above.

Commands run: listed above.

Proof objects produced: listed above.

Unresolved risks: 20260509 step-11 `MATH:coftz` theta explosion remains.

Next decision needed: dispatch the next focused investigation at the
20260509 physics/vertical-implicit `coftz` failure, using the passing 2/5/10
controlled parity proof here as a boundary around the shared acoustic core.
