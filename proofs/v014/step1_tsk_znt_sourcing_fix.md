# V0.14 Step-1 TSK/ZNT Sourcing Fix

Verdict: `TSK_ZNT_THERMO_INPUTS_FIXED_NEXT_BLOCKER_SURFACE_LAYER_OUTPUTS`.

## WRF-Anchored Result

- `TSK` at `SFCLAY_mynn` input: max_abs `0.0` K.
- `ZNT` at `SFCLAY_mynn` input: max_abs `1.1920928910669204e-08` m, RMSE `8.108127886677328e-10`.
- `MAVAIL` at `SFCLAY_mynn` input: max_abs `1.1920928966180355e-08`.
- WRF `wrfinput_d02` has no direct `ZNT`/`Z0`; WRF initializes `ZNT` from `LANDUSE.TBL` `SFZ0/100` by `LU_INDEX` before this call.
- Old roughness surrogate witness: max_abs `0.7737602195739746` m; table-backed source max_abs `1.1920928910669204e-08` m.

## Remaining Blocker

TSK/ZNT and the non-surface thermodynamic column entering `SFCLAY_mynn` are fixed/bounded; the next WRF-anchored blocker is now the surface-layer output algebra:

- `th_phy(kts)` max_abs `6.71089752017906e-05` K, RMSE `1.3430183262692343e-05`.
- derived `t_phy(kts)` max_abs `0.013577942721781255` K, RMSE `0.0010959870065792568`.
- `p_phy(kts)` max_abs `0.015625` Pa, RMSE `0.0013253267749381015`.
- `u/v/qv(kts)` are bounded at max_abs `7.152557373046875e-07`, `7.152557373046875e-07`, `4.9845645343910006e-08`.

Surface output remains red with exact TSK/ZNT/MAVAIL:

- `UST` max_abs `0.01231782267117762`, RMSE `0.0007831552182476275`.
- `HFX` max_abs `27.09163832864155`, RMSE `1.3972156996573475`.
- `QFX` max_abs `2.744275103194571e-07`, RMSE `8.62190066790179e-08`.

## Strict Step-1

- after-conv `T_TENDF` max_abs `847.1445725702908`, RMSE `9.56593990212596`.

## Files

- JSON proof: `/home/enric/src/wrf_gpu2/proofs/v014/step1_tsk_znt_sourcing_fix.json`
- WRF hook patch archive: `/home/enric/src/wrf_gpu2/proofs/v014/step1_tsk_znt_sourcing_fix_wrf_patch.diff`
