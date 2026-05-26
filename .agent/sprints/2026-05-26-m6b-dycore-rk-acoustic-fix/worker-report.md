# Worker Report

Summary: BLOCKED - partial V-tendency fix landed, but sprint acceptance is not fully satisfied.

## Summary

I fixed the named step-46 V runaway in `dycore_rk_acoustic` by suppressing the unvalidated reduced M4 V self-advection inside the M6b operational RK/acoustic wrapper while preserving resident/base V tendencies. The fixed localizer now reports `dycore_rk_acoustic = 0.0 m/s/s` at the old bad decomposition point and step-46 max |V| is 11.480101585388184 m/s instead of 103.72041320800781 m/s.

The sprint remains blocked because broader required validation does not pass: the 360-step audit first fails at step 49 on theta upper bound and later nonfinites, the real-IC multi-step parity probe fails from step 2 in theta/t_2ave/ph_tend while U/V remain bitwise equal, and `pytest -x` fails on a missing external fixture file.

## Files Changed

- `src/gpuwrf/runtime/operational_mode.py`
- `tests/test_m6b_dycore_rk_acoustic_fix.py`
- `.agent/sprints/2026-05-26-m6b-dycore-rk-acoustic-fix/proof_baseline_reproduces.json`
- `.agent/sprints/2026-05-26-m6b-dycore-rk-acoustic-fix/hypothesis_notes.md`
- `.agent/sprints/2026-05-26-m6b-dycore-rk-acoustic-fix/proof_fix_validation.json`
- `.agent/sprints/2026-05-26-m6b-dycore-rk-acoustic-fix/proof_360_step_bounds.json`
- `.agent/sprints/2026-05-26-m6b-dycore-rk-acoustic-fix/baseline/*`
- `.agent/sprints/2026-05-26-m6b-dycore-rk-acoustic-fix/fixed/*`

## Commands Run

- `taskset -c 0-3 python scripts/m6b_v3_localize_521.py --run-id 20260521_18z_l3_24h_20260522T072630Z --output .agent/sprints/2026-05-26-m6b-dycore-rk-acoustic-fix/baseline/`
  - Exit 0. Output: `status=OK`, `verdict=NAMED-FIX:dycore_rk_acoustic`. Baseline reproduced step 46 |V|max 103.72041320800781 m/s at [33,53,36].
- `taskset -c 0-3 python scripts/m6b_v3_localize_521.py --run-id 20260521_18z_l3_24h_20260522T072630Z --output .agent/sprints/2026-05-26-m6b-dycore-rk-acoustic-fix/fixed/`
  - Exit 0. Output: `status=OK`, `verdict=NAMED-FIX:boundary_application`; fixed step 46 passed wind bounds and `dycore_rk_acoustic` contribution was 0.0.
- `taskset -c 0-3 python scripts/m6b6_coupled_step_compare.py`
  - Exit 2. Output: `--tier is required unless --synthetic-dryrun is set`.
- `taskset -c 0-3 python scripts/m6b6_coupled_step_compare.py --tier all`
  - Exit 0. Output head/tail: `passed: true`, `outcome: SEVENTH-COUPLED-STEP-PARITY-ACHIEVED`, `diverging_field_count: 0`.
- `taskset -c 0-3 python scripts/m6b_real_ic_operational_compare.py --steps 2,5,10`
  - Exit 2. Output: `argument --steps: invalid int value: '2,5,10'`.
- `taskset -c 0-3 python scripts/m6b_real_ic_operational_compare.py --steps 2`
  - Exit 2. Output: step 1 max_abs_delta 0.0; step 2 nonfinite theta/t_2ave/ph_tend, V max_abs_delta 0.0.
- `taskset -c 0-3 python scripts/m6b_real_ic_operational_compare.py --steps 5`
  - Exit 2. Output: same step-2 theta/t_2ave/ph_tend nonfinite failure; V remains 0.0 delta in reported fields.
- `taskset -c 0-3 python scripts/m6b_real_ic_operational_compare.py --steps 10`
  - Exit 2. Output: same step-2 theta/t_2ave/ph_tend nonfinite failure; V remains 0.0 delta in reported fields.
- Custom 360-step audit using `scripts/m6b_v3_localize_521.py` internals.
  - Exit 0, proof status FAIL. First bad step 49: theta_upper_14_max_k 1343.5467529296875; V max before nonfinites 11.480101585388184 m/s.
- `taskset -c 0-3 pytest -x`
  - Exit 1. Output: `tests/test_canary_wrf_fixture.py::test_full_external_file_exists_at_external_uri` failed because `data/fixtures/canary-wrf-d01-20260518T18-tslice-v1/full.npz` is missing.
- `taskset -c 0-3 pytest tests/test_m6b_dycore_rk_acoustic_fix.py -v`
  - Exit 0. Output: `1 passed`.

## Proof Objects

- `.agent/sprints/2026-05-26-m6b-dycore-rk-acoustic-fix/proof_baseline_reproduces.json`
- `.agent/sprints/2026-05-26-m6b-dycore-rk-acoustic-fix/proof_fix_validation.json`
- `.agent/sprints/2026-05-26-m6b-dycore-rk-acoustic-fix/proof_360_step_bounds.json`
- `.agent/sprints/2026-05-26-m6b-dycore-rk-acoustic-fix/baseline/proof_step46_violation.json`
- `.agent/sprints/2026-05-26-m6b-dycore-rk-acoustic-fix/fixed/proof_step46_violation.json`
- `.agent/sprints/2026-05-26-m6b-dycore-rk-acoustic-fix/fixed/proof_operator_decomposition.json`

## Risks

- The fix is intentionally narrow and operational-only; it does not claim a complete WRF `advance_uv` implementation.
- Full 1h acceptance remains blocked by thermodynamic instability after the V runaway is removed.
- Existing fixture data is missing in this worktree, so full pytest cannot be green here.
- Two listed validation commands have stale CLI syntax relative to the checked-in scripts.

## Handoff

Objective: fix the step-46 V runaway attributed to `dycore_rk_acoustic`.

Files changed: see above.

Commands run: see above.

Proof objects produced: see above.

Unresolved risks: 360-step theta/nonfinite failure, multi-step theta parity failure, missing external fixture.

Next decision needed: manager should either accept this as a partial V fix and dispatch a thermodynamic stability/parity follow-up, or reject the operational V-advection suppression as too broad for M6b.
