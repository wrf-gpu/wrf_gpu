# Worker Report

Summary: BLOCKED_ACCEPTANCE_NOT_MET. I implemented the smallest allowed `coftz` source fix in the contract-owned MPAS-style vertical path: `build_epssm_column_coefficients` now accepts an optional stable theta coefficient source, and `_mpas_recurrence_vertical_update` passes `theta_base` so `coftz` is not built from instantaneous runaway theta. The focused regression passes, but the required 20260509 operational replay is unchanged. Evidence points to the current operational path exercising `dynamics/core` shared acoustic recurrence, which this sprint marks read-only.

## Files Changed

- `src/gpuwrf/dynamics/vertical_implicit_solver.py`
- `src/gpuwrf/dynamics/acoustic_wrf.py`
- `tests/test_m6b_coftz_theta_fix.py`
- `.agent/sprints/2026-05-26-m6b-coftz-theta-fix/hypothesis_notes.md`
- `.agent/sprints/2026-05-26-m6b-coftz-theta-fix/proof_baseline_reproduces.json`
- `.agent/sprints/2026-05-26-m6b-coftz-theta-fix/proof_fix_validation.json`
- `.agent/sprints/2026-05-26-m6b-coftz-theta-fix/baseline/*`
- `.agent/sprints/2026-05-26-m6b-coftz-theta-fix/fixed/*`

## Commands Run + Output

- `taskset -c 0-3 python scripts/m6b_v3_localize_509.py --run-id 20260509_18z_l3_24h_20260511T190519Z --output .../baseline/`
  - Output: status `MATH:coftz`; baseline reproduced `THETA_BOUND_VIOLATION` at step 11 / lead 110 s, k=28 j=60 i=73, theta `2604313608192.0 K`.
- `pytest tests/test_m6b_coftz_theta_fix.py -v`
  - Output: `2 passed`.
- `python scripts/m6b6_coupled_step_compare.py`
  - Output: failed CLI parse, `--tier is required unless --synthetic-dryrun is set`.
- `python scripts/m6b6_coupled_step_compare.py --tier all`
  - Output: stopped after several minutes with no stdout; replacement command below used for supported proof.
- `python scripts/m6b6_coupled_step_compare.py --synthetic-dryrun`
  - Output: `passed: true`, `clean_self_compare_passed: true`, boundary perturbations caught.
- `python scripts/m6b_real_ic_operational_compare.py --steps 2,5,10`
  - Output: failed CLI parse, `invalid int value: '2,5,10'`.
- `python scripts/m6b_real_ic_operational_compare.py --steps 2`, `--steps 5`, `--steps 10`
  - Output: all `status: FAIL`; step 1 exact, from step 2 `theta`, `t_2ave`, and `ph_tend` have `max_abs_delta: 1e+300` / nonfinite.
- `taskset -c 0-3 python scripts/m6b_v3_localize_509.py --run-id 20260509_18z_l3_24h_20260511T190519Z --output .../fixed/`
  - Output: status still `MATH:coftz`; same step-11 theta violation as baseline.
- `taskset -c 0-3 pytest -x`
  - Output: failed at `tests/test_canary_wrf_fixture.py::test_full_external_file_exists_at_external_uri`; missing `data/fixtures/canary-wrf-d01-20260518T18-tslice-v1/full.npz`.
- `taskset -c 0-3 pytest tests/test_m6b_coftz_theta_fix.py -v`
  - Output: `2 passed`.

## Proof Objects

- `.agent/sprints/2026-05-26-m6b-coftz-theta-fix/proof_baseline_reproduces.json`
- `.agent/sprints/2026-05-26-m6b-coftz-theta-fix/proof_fix_validation.json`
- `.agent/sprints/2026-05-26-m6b-coftz-theta-fix/baseline/proof_theta_explosion.json`
- `.agent/sprints/2026-05-26-m6b-coftz-theta-fix/fixed/proof_theta_explosion.json`
- `.agent/sprints/2026-05-26-m6b-coftz-theta-fix/hypothesis_notes.md`

## Risks

- The patch is correct for the contract-owned `vertical_implicit_solver.py` / `_mpas_recurrence_vertical_update` path, but it does not affect the current operational replay.
- Full acceptance requires changing or routing through `dynamics/core` acoustic recurrence, which is read-only in this sprint.
- `pytest -x` is also blocked by a missing external fixture file under `data/fixtures`.

## Handoff

Objective: fix the 20260509 `coftz` theta explosion. Result: blocked; nearest-fit hypothesis is H2 (`theta_face` source mismatch), but the operational code path needing the fix is outside allowed ownership. Next decision needed: manager should either expand ownership to `src/gpuwrf/dynamics/core/*` / operational routing or dispatch a new sprint specifically for the shared acoustic core theta/t_2ave/ph_tend step-2 failure.
