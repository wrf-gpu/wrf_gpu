# Manager Closeout

## Outcome

Accepted. The sprint fixes the `PSFC` vapor-light diagnostic bug with a
WRF-anchored formula and keeps the deeper 3D pressure-state issue visible as a
separate blocker. The old Canary LBCFIX 72h run was stopped after h24 because it
used the old formula and is characterization only.

## Proof Objects

- `proofs/v014/psfc_moist_pressure_state_closure.{py,json,md}`
- `.agent/reviews/2026-06-10-v014-fable-psfc-moist-pressure-closure.md`
- `/mnt/data/wrf_gpu_validation/v014_canary_d02_72h_lbcfix_20260610T151455Z/canary_d02_h24_grid_compare.md`

## Merge Decision

Merge Decision:

ACCEPT_AND_MERGE_AFTER_CPU_GATES. The patch is diagnostic-only and WRF-source
anchored; it materially lowers expected `PSFC` error and does not hide the
remaining `P/PH` dynamics lane.

## Scope Changes

The next release blocker is now explicit: moist `cqw` / moist `pg_buoy_w`
threading for the acoustic w-equation, with savepoint/oracle and GPU gates.
Switzerland GPU remains blocked until this lane is closed or formally bounded.

## Lessons

All-cell field parity plus pressure-budget proofs found a release-relevant
diagnostic mistake that station TOST would not have isolated. WRF runtime
diagnostics must be source-anchored before being used as grid-parity gates.

## Next Sprint

Run short GPU h1/h4 validation for this PSFC diagnostic fix, then open the
moist-cqw dynamics sprint for the 3D pressure-state lane.
