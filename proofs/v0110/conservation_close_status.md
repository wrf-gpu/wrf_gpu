# v0.11.0 Conservation Close Status

- Budgets closed: True.
- Dry-mass relative residual: 0.000000e+00.
- Total-water relative residual: 0.000000e+00.
- Moist-static-energy relative residual: 0.000000e+00.
- Post-boundary guard changed cells: 0.
- Post-boundary nonfinite replacements: 0.

## What Changed

- Dry physics deltas are converted into RK1-fixed `DryPhysicsTendencies` and passed through `rk_addtend_dry` at each RK stage.
- Non-dry physics prognostics are applied as physics deltas after the dycore so qv advection is not overwritten.
- MYNN scale-aware PBL-height input now uses the same positive PBLH floor as the option-1 length path, removing the marine-column NaN source before boundary guards.

## Evidence Notes

- The proof records raw `post_boundary_raw` finite summaries before any post-boundary finite/origin guard.
- RK/dycore, non-dry physics, and lateral-boundary source corrections are phase-accounted in the JSON; they are not nonfinite replacement masks.
