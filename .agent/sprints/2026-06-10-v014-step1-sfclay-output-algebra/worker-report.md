# Worker Report

Summary: The sprint fixed three local WRF `SFCLAY1D_mynn` algebra mismatches
and narrowed the active Step-1 blocker to later MYNN/PBL source coupling after
the fixed surface outputs.

## Objective

Close or strictly narrow the MYNN surface-layer output mismatch after
`TSK/ZNT/MAVAIL` and WRF `phy_prep` thermodynamic inputs were fixed at the
`sfclay_mynn` hook.

## Files Changed

- `src/gpuwrf/physics/surface_layer.py`
- `src/gpuwrf/coupling/physics_couplers.py`
- `tests/test_v014_mynn_surface_layer_regressions.py`
- `proofs/v014/step1_sfclay_output_algebra.{py,json,md}`
- `proofs/v014/step1_sfclay_output_algebra_wrf_patch.diff`
- `.agent/reviews/2026-06-10-v014-step1-sfclay-output-algebra.md`
- refreshed v0.14 Step-1 proof artifacts from required reruns.

## Result

The local surface-layer output gap was caused by WRF details missing in the JAX
surface path: first-timestep MYNN bulk-Richardson clamp, `QVSH` specific
humidity in virtual-theta terms, and WRF `phy_prep` density
`rho=(1+QVAPOR)/ALT` rather than ideal-gas density from surface pressure.

## Key Evidence

- `UST` max_abs `0.0007252174862408534`, RMSE `1.53999402707944e-05`.
- `HFX` max_abs `0.2643125302157898`, RMSE `0.022548398654638105`.
- `QFX` max_abs `6.468560998136325e-08`, RMSE `3.002727253934746e-08`.
- `BR` max_abs `0.01166976922050278`, RMSE `0.0003583716190119449`.
- `rho` max_abs `0.00018143653869628906`, RMSE `7.786468426065368e-06`.
- strict after-conv `T_TENDF` remains red at max_abs
  `847.1446969755725`, RMSE `9.627208432391289`.

## Remaining Risk

Strict Step-1 is still red. The new WRF-anchored blocker is later than
`SFCLAY1D_mynn` output algebra: MYNN/PBL source coupling after the fixed surface
outputs.
