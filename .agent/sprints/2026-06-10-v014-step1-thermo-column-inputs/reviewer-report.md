# Reviewer Report

Decision: ACCEPT_AS_LOCAL_CORRECTNESS_FIX.

The patch is scoped to the grid-backed surface-column boundary and keeps the
old metric-free fallback for analytic callers. It is WRF-sourced rather than a
tolerance adjustment: the proof anchors the conversions to WRF `phy_prep`,
`surface_driver`, and the existing exact `sfclay_mynn` hook.

## Review Notes

- The `float32` hydrostatic reconstruction is intentional because the WRF
  physics-prep pressure path is single-real; the proof shows this closes
  `p_phy` to `0.015625 Pa`.
- Passing `grid` through operational/grid-bearing call sites prevents this fix
  from being limited to only one proof harness.
- The new `_SurfaceColumnState.t_air` and `.psfc` fields are optional, so older
  no-grid callers remain compatible.
- The strict Step-1 residual is still large, so the sprint correctly narrows
  rather than closes parity.

## Required Follow-Up

Open the next GPT-5.5 xhigh sprint around WRF internals inside
`module_sf_mynn.F` / `SFCLAY1D_mynn`, especially
`thx/thgb/br/zol/psim/psih/ust/hfx/qfx`.
