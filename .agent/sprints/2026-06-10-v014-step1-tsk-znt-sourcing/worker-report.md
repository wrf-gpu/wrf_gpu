# Worker Report

Summary: The sprint fixed the WRF-anchored Step-1 `TSK/ZNT/MAVAIL` sourcing
boundary and reduced the active blocker to non-surface thermodynamic column
inputs entering `sfclay_mynn`.

## Objective

Prove or refute whether the remaining Step-1 surface mismatch after the
first-call surface semantics fix was caused by `TSK/ZNT` input sourcing. If
local, fix production sourcing and rerun the source-fidelity proofs.

## Files Changed

- `src/gpuwrf/physics/noah_mp.py`
- `src/gpuwrf/io/land_state.py`
- `tests/test_m6_noah_mp_prescribed.py`
- `tests/savepoint/test_static_fields.py`
- `proofs/v014/step1_tsk_znt_sourcing_fix.py`
- `proofs/v014/step1_tsk_znt_sourcing_fix.{json,md}`
- `proofs/v014/step1_tsk_znt_sourcing_fix_wrf_patch.diff`
- refreshed Step-1/MYNN proof summaries.

## Result

WRF `wrfinput_d02` lacks direct `ZNT/Z0`, so WRF cold-starts `ZNT` from
`LANDUSE.TBL` `SFZ0/100` by `LU_INDEX`. The JAX path now mirrors this for the
MODIFIED_IGBP_MODIS_NOAH table, and also mirrors `MAVAIL` via `SLMO`.

Proof verdict: `TSK_ZNT_SOURCE_FIXED_NEXT_BLOCKER_THERMODYNAMIC_COLUMN_INPUTS`.

## Key Evidence

- `TSK` at the exact `sfclay_mynn` input hook: max_abs `0.0 K`.
- `ZNT` at the hook: max_abs `1.1920928910669204e-08 m`.
- `MAVAIL` at the hook: max_abs `1.1920928966180355e-08`.
- strict after-conv `T_TENDF` remains red: max_abs `1497.6112467075195`, RMSE
  `13.252694871222973`.

## Remaining Risk

Surface outputs remain red because `th_phy(kts)`, derived `t_phy(kts)`, and
`p_phy(kts)` differ before `sfclay_mynn`. That is the next boundary, not a
TSK/ZNT issue.
