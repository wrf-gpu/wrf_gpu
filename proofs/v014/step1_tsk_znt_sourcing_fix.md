# V0.14 Step-1 TSK/ZNT Sourcing Fix

Verdict: `TSK_ZNT_SOURCE_FIXED_NEXT_BLOCKER_THERMODYNAMIC_COLUMN_INPUTS`.

## WRF-Anchored Result

- `TSK` at `SFCLAY_mynn` input: max_abs `0.0` K.
- `ZNT` at `SFCLAY_mynn` input: max_abs `1.1920928910669204e-08` m, RMSE `8.108127886677328e-10`.
- `MAVAIL` at `SFCLAY_mynn` input: max_abs `1.1920928966180355e-08`.
- WRF `wrfinput_d02` has no direct `ZNT`/`Z0`; WRF initializes `ZNT` from `LANDUSE.TBL` `SFZ0/100` by `LU_INDEX` before this call.
- Old roughness surrogate witness: max_abs `0.7737602195739746` m; table-backed source max_abs `1.1920928910669204e-08` m.

## Remaining Blocker

TSK/ZNT source is no longer the Step-1 blocker. The next WRF-anchored blocker is the non-surface thermodynamic column entering `SFCLAY_mynn`:

- `th_phy(kts)` max_abs `5.490148027499686` K, RMSE `4.596847297193302`.
- derived `t_phy(kts)` max_abs `5.521345498302992` K, RMSE `4.614221839008816`.
- `p_phy(kts)` max_abs `292.8203125` Pa, RMSE `45.931279429597595`.
- `u/v/qv(kts)` are bounded at max_abs `7.152557373046875e-07`, `7.152557373046875e-07`, `4.9845645343910006e-08`.

Surface output remains red with exact TSK/ZNT/MAVAIL:

- `UST` max_abs `0.14784556445623043`, RMSE `0.028969744732629553`.
- `HFX` max_abs `160.6217929042739`, RMSE `21.187735409878695`.
- `QFX` max_abs `2.7273259067191847e-05`, RMSE `1.745930070321283e-05`.

## Strict Step-1

- after-conv `T_TENDF` max_abs `1497.6112467075195`, RMSE `13.252694871222973`.

## Files

- JSON proof: `/home/enric/src/wrf_gpu2/proofs/v014/step1_tsk_znt_sourcing_fix.json`
- WRF hook patch archive: `/home/enric/src/wrf_gpu2/proofs/v014/step1_tsk_znt_sourcing_fix_wrf_patch.diff`
