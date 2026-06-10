# Review: V0.14 Step-1 TSK/ZNT Sourcing

Verdict: `TSK_ZNT_SOURCE_FIXED_NEXT_BLOCKER_THERMODYNAMIC_COLUMN_INPUTS`.

Pre-sfclay `TSK` is exact: max_abs `0.0`.
Pre-sfclay `ZNT` is fixed: max_abs `1.1920928910669204e-08`.
Strict Step-1 remains red: max_abs `1497.6112467075195`, RMSE `13.252694871222973`.

Next blocker: non-TSK/ZNT thermodynamic column inputs at `SFCLAY_mynn`.
`th_phy(kts)` max_abs `5.490148027499686`; `t_phy(kts)` max_abs `5.521345498302992`; `p_phy(kts)` max_abs `292.8203125`.

Proof: `/home/enric/src/wrf_gpu2/proofs/v014/step1_tsk_znt_sourcing_fix.md`
