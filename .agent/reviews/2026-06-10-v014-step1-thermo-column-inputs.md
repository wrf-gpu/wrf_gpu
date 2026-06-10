# Review: V0.14 Step-1 Thermodynamic Column Inputs

Verdict: `THERMO_COLUMN_INPUTS_FIXED_NEXT_BLOCKER_SURFACE_LAYER_OUTPUTS`.

The prior `th_phy/t_phy/p_phy/dz8w` blocker is local and fixed in the grid-backed `_surface_column_view`: dry theta_m conversion, WRF hydrostatic `p_hyd`/`psfc`, WRF `g=9.81` dz, and explicit `t_air` for WRF's split `t_phy` semantics.

Fixed maxima: `th_phy` `6.71089752017906e-05` K; `t_phy` `0.013577942721781255` K; `p_phy` `0.015625` Pa; `dz8w` `0.00018988715282830526` m.

Strict Step-1 remains red: max_abs `847.1445725702908`, RMSE `9.56593990212596`.
Next blocker is later surface-layer output algebra: `UST` max_abs `0.01231782267117762`, `HFX` max_abs `27.09163832864155`.

Proof: `/home/enric/src/wrf_gpu2/proofs/v014/step1_thermo_column_inputs.md`
