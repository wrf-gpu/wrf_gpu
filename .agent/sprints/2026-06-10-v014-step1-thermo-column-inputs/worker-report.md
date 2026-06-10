# Worker Report

Summary: The sprint fixed the WRF `phy_prep` thermodynamic input boundary for
the grid-backed surface path and narrowed the active Step-1 blocker to later
`module_sf_mynn` output algebra.

## Objective

Close or strictly narrow the `sfclay_mynn` `th_phy/t_phy/p_phy/dz8w` mismatch
after TSK/ZNT/MAVAIL sourcing was fixed.

## Files Changed

- `src/gpuwrf/coupling/physics_couplers.py`
- `src/gpuwrf/coupling/driver.py`
- `src/gpuwrf/diagnostics/comprehensive_harness.py`
- `src/gpuwrf/integration/d02_replay.py`
- `src/gpuwrf/runtime/operational_mode.py`
- `tests/test_v014_dry_source_leaf_wiring.py`
- `proofs/v014/step1_thermo_column_inputs.{py,json,md}`
- refreshed v0.14 Step-1 proof artifacts.

## Result

The root cause was local to the surface-column view. WRF `phy_prep` does not
feed the surface layer with raw live-state `theta_m`, nonhydrostatic `state.p`,
or standard-gravity `dz`. The grid-backed JAX view now provides WRF dry
`th_phy`, hydrostatic `p_hyd`/`psfc`, explicit `t_air`, and WRF `g=9.81` `dz8w`.

## Key Evidence

- `th_phy(kts)` max_abs improved `5.490148027499686 K -> 6.71089752017906e-05 K`.
- `p_phy(kts)` max_abs improved `292.8203125 Pa -> 0.015625 Pa`.
- `dz8w(kts)` max_abs is `0.00018988715282830526 m`.
- strict after-conv `T_TENDF` improved to max_abs `847.1445725702908`, RMSE
  `9.56593990212596`.

## Remaining Risk

Strict Step-1 is still red. The next exact blocker is later surface-layer output
algebra inside `module_sf_mynn` after the fixed input tuple.
