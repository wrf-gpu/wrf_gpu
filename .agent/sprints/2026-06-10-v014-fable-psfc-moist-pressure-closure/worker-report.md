# Worker Report

## Summary

Summary:

Fable high closed the fixed-Canary `PSFC` vapor-light diagnostic bug with a
WRF-anchored production fix and formally bounded the deeper 3D pressure-state
lane as a separate moist-cqw dynamics sprint. The fix changes `PSFC` from a
height extrapolation of nonhydrostatic `P+PB` to WRF runtime `p_hyd_w(kts)`, the
moist hydrostatic integral over dry-mass coordinates.

## Files Changed

- `src/gpuwrf/runtime/operational_mode.py`
- `src/gpuwrf/io/wrfout_writer.py`
- `scripts/diag/d03_pressure_knockout.py`
- `tests/test_v014_psfc_moist_hydrostatic.py`
- `proofs/v014/psfc_moist_pressure_state_closure.{py,json,md}`
- `.agent/reviews/2026-06-10-v014-fable-psfc-moist-pressure-closure.md`

## Commands Run

See the review artifact for the worker-run command list. The manager reran the
proof, JSON validation, new tests, focused writer/pipeline tests, compileall,
and `git diff --check`.

## Proof Objects

- `proofs/v014/psfc_moist_pressure_state_closure.py`
- `proofs/v014/psfc_moist_pressure_state_closure.json`
- `proofs/v014/psfc_moist_pressure_state_closure.md`
- `.agent/reviews/2026-06-10-v014-fable-psfc-moist-pressure-closure.md`

## Risks

The fix is diagnostic-only for `PSFC`. The 3D `P/PH/W` pressure-state lane
remains dry-balanced and needs the follow-up moist-cqw dynamics sprint before
field-parity promotion.

## Handoff

Merge the diagnostic fix after manager review, then run a short GPU h1/h4
validation from the fixed code. Do not resume Switzerland GPU until the 3D
moist-cqw pressure-state lane is closed or formally bounded.
