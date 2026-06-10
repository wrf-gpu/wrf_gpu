# Worker Report

Summary: GPT fixed WRF MYNN surface-layer first-call semantics and narrowed the
remaining Step-1 blocker to TSK/ZNT surface input sourcing.

## Objective

Close, or reduce to one strictly narrower WRF-anchored blocker, the Step-1
surface-layer flux/input boundary divergence feeding MYNN.

## Files Changed

- `src/gpuwrf/physics/surface_layer.py`
- `src/gpuwrf/coupling/physics_couplers.py`
- `src/gpuwrf/integration/d02_replay.py`
- `src/gpuwrf/runtime/operational_mode.py`
- `tests/test_m6_surface_layer_kernel.py`
- `proofs/v014/step1_sfclay_boundary_fix.py`
- `proofs/v014/step1_sfclay_boundary_fix.json`
- `proofs/v014/step1_sfclay_boundary_fix.md`
- refreshed MYNN/source proof artifacts and reviews.

## Outcome

Verdict:

`STEP1_SFCLAY_FIRST_CALL_FIXED_NEXT_BLOCKER_TSK_ZNT_SURFACE_INPUTS`

Implemented WRF MYNN surface first-call semantics:

- UST first guess: `max(0.04 * sqrt(u^2 + v^2), 0.001)`.
- `MOL=0` first-call behavior.
- land `QSFC=qv/(1+qv)` carried value.
- Li_etal_2010 z/L seed for first-call `zolrib`.

The strict Step-1 residual remains red. The next exact blocker is TSK/ZNT input
sourcing before `sfclay_mynn`.

## Proof Summary

- first-call land QSFC max_abs `0.0`.
- UST RMSE improved `0.08667703917523994 -> 0.02954126268295198`.
- qv-flux RMSE improved `1.9833425562981398e-05 -> 1.442591864492997e-05`.
- strict after-conv `T_TENDF` max_abs `1497.6112467075195`, RMSE
  `13.296448784742802`.
- surviving TSK max_abs `8.344940187890643 K`; ZNT max_abs
  `0.9737602076530456 m`.

## Commands Run

Worker reported py_compile, focused surface tests, JSON validation, `git diff
--check`, `step1_sfclay_boundary_fix.py`, `mynn_driver_source_output_fix.py`,
and `step1_source_fidelity_closure.py`.

## Next Boundary

Emit a tiny WRF Step-1 surface-driver hook around
`module_surface_driver/module_sf_mynn` for incoming
`TSK/ZNT/UST/QSFC/MOL` and outgoing `UST/HFX/QFX/ZNT`, compare against JAX
`_surface_column_view` inputs and diagnostics, then fix TSK/ZNT sourcing if
confirmed.
