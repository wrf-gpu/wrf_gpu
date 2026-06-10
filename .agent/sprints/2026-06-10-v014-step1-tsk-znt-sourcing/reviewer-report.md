# Reviewer Report

Decision: ACCEPT_AS_CORRECT_NARROWING_FIX.

The implementation is scoped to lower-boundary cold-start sourcing and avoids
dycore, GPU runtime, memory, or FP32 changes. The WRF hook evidence is strong
enough for this boundary: it shows the old roughness surrogate was wrong by up
to `0.7737602195739746 m`, while the new table-backed source is within
`1.1920928910669204e-08 m` at the exact `sfclay_mynn` input.

## Review Notes

- The production table is explicitly the MODIFIED_IGBP_MODIS_NOAH Noah table,
  matching the current fixture path. This is acceptable for v0.14's current
  validation case; future generalized land-use dispatch should be a separate
  fail-closed extension, not a blocker for this fix.
- The fallback path remains available when `LU_INDEX` is unavailable, so older
  smoke callers are not forced into a new table assumption.
- No host/device transfer, dynamic-shape allocation, or GPU-only behavior was
  introduced.
- The proof correctly does not overclaim: surface fluxes and strict Step-1 are
  still red.

## Required Follow-Up

Open the next GPT-5.5 xhigh sprint around the exact `sfclay_mynn`
thermodynamic-column boundary: `th_phy(kts)`, `t_phy(kts)`, `p_phy(kts)`, and
`dz8w` against JAX `_surface_column_view`.
