# Review: V0.14 Step-1 TSK/ZNT Sourcing

Verdict: `TSK_ZNT_THERMO_INPUTS_FIXED_NEXT_BLOCKER_SURFACE_LAYER_OUTPUTS`.

Pre-sfclay `TSK` is exact: max_abs `0.0`.
Pre-sfclay `ZNT` is fixed: max_abs `1.1920928910669204e-08`.
Strict Step-1 remains red: max_abs `847.1445725702908`, RMSE `9.56593990212596`.

Next blocker: surface-layer outputs after fixed TSK/ZNT/thermodynamic inputs.
`th_phy(kts)` max_abs `6.71089752017906e-05`; `t_phy(kts)` max_abs `0.013577942721781255`; `p_phy(kts)` max_abs `0.015625`.

Proof: `/home/enric/src/wrf_gpu2/proofs/v014/step1_tsk_znt_sourcing_fix.md`
