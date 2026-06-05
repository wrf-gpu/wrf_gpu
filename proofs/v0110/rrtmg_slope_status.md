# v0.11.0 RRTMG slope-radiation status

- status: PARTIAL
- radiation parity: PASS
- fixture: `/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z/wrfout_d02_2026-05-22_12:00:00`
- selected columns: 16 largest terrain-radiation deltas
- gross SWDOWN RMSE vs WRF SWDNB: 7.561 W m-2
- topo SWNORM RMSE vs WRF SWNORM: 6.693 W m-2
- GLW RMSE vs WRF GLW: 2.310 W m-2
- selected shadow-mask count: 0
- shadow note: this d02 daytime fixture exercises slope/aspect; computed WRF-style ray shadows are zero in the selected columns.

Runtime wiring: real XLAT/XLONG radiation static fields are on the operational namelist; `topo_shading=1` and `slope_rad=1` are read from the Canary WRF namelist; Noah-MP `albedo`/`emiss` are used over land when a Noah-MP land carry is present.

Short d02 GPU sanity: `proofs/v0110/rrtmg_slope_gpu_sanity.json` is PARTIAL. One real d02 physics step with radiation enabled produced interior non-finites in theta/u/v, while the post-step land/daylight RRTMG fields were finite and the terrain SW signal was visible (topographic delta -84.7 to +42.5 W m-2; correction 0.9186 to 1.0406). A topo-off control (`topo_shading=0`, `slope_rad=0`) produced the same one-step non-finite counts, so this is not specific to the topo-shading/slope-radiation wiring. Carry this full-physics one-step stability issue to the manager/Opus second-line debugger if it is required as a milestone close gate.
