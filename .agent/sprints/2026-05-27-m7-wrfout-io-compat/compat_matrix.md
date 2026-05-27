# M7 wrfout I/O Compatibility Matrix

Generated UTC: 2026-05-27T00:15:01+00:00
CPU reference: `/mnt/data/canairy_meteo/runs/wrf_l3/20260525_18z_l3_24h_20260526T221207Z/wrfout_d02_2026-05-25_18:00:00`
GPU writer: `/tmp/wrf_gpu2_iocompat/src/gpuwrf/coupling/driver.py:1139`

## Summary

- CPU WRF variables: 375
- GPU payload keys: 20
- Classification counts: {'DEVIATION_DOCUMENTED': 13, 'EXTRA_GPU': 7, 'MISSING_GPU': 362}
- Downstream-consumed classification counts: {'DEVIATION_DOCUMENTED': 5, 'MISSING_GPU': 21}
- Compatibility verdict: structural audit complete; GPU output is not yet a drop-in NetCDF wrfout.

## Downstream-Critical Rows

| Variable | Classification | CPU has | GPU writes | Notes |
|---|---:|---:|---:|---|
| `CLDFRA` | MISSING_GPU | YES | NO | Cloud products; thin extractor derives TCC/CLDLOW/CLDMID/CLDHIGH. |
| `GLW` | MISSING_GPU | YES | NO | Radiation/cloud feature and point-shadow products. |
| `HFX` | MISSING_GPU | YES | NO | Surface heat-flux diagnostics. |
| `HGT` | MISSING_GPU | YES | NO | Terrain/georef audits and pressure-derived product proxies. |
| `LANDMASK` | MISSING_GPU | YES | NO | Surface/geography and cloud/terrain diagnostics. |
| `LH` | MISSING_GPU | YES | NO | Latent heat-flux diagnostics. |
| `LU_INDEX` | MISSING_GPU | YES | NO | Surface/geography diagnostics. |
| `PBLH` | MISSING_GPU | YES | NO | PBL diagnostics and point-shadow products. |
| `PSFC` | MISSING_GPU | YES | NO | Surface pressure, QA, and MSLP proxy input. |
| `Q2` | DEVIATION_DOCUMENTED | YES | YES | Surface diagnostic present; Time dimension and NetCDF attributes omitted. |
| `QCLOUD` | MISSING_GPU | YES | NO | Cloud-cap diagnostics and cloud feature products. |
| `QICE` | MISSING_GPU | YES | NO | Cloud-cap diagnostics and cloud feature products. |
| `QRAIN` | MISSING_GPU | YES | NO | Cloud/precip diagnostics. |
| `RAINC` | MISSING_GPU | YES | NO | Convective accumulated precipitation for precip verification. |
| `RAINNC` | MISSING_GPU | YES | NO | Grid-scale accumulated precipitation for precip verification. |
| `RAINSH` | MISSING_GPU | YES | NO | Shallow-convective accumulated precipitation for precip verification. |
| `SWDOWN` | MISSING_GPU | YES | NO | Solar/cloud feature and point-shadow products. |
| `T2` | DEVIATION_DOCUMENTED | YES | YES | Surface diagnostic present; Time dimension and NetCDF attributes omitted. |
| `TSK` | MISSING_GPU | YES | NO | Skin-temperature diagnostics and surface-state QA. |
| `Times` | MISSING_GPU | YES | NO | WRF time coordinate used by raw wrfout and thin-NetCDF consumers. |
| `U10` | DEVIATION_DOCUMENTED | YES | YES | Surface diagnostic present; Time dimension and NetCDF attributes omitted. |
| `UST` | DEVIATION_DOCUMENTED | YES | YES | Surface friction velocity present; Time dimension and NetCDF attributes omitted. |
| `V10` | DEVIATION_DOCUMENTED | YES | YES | Surface diagnostic present; Time dimension and NetCDF attributes omitted. |
| `XLAT` | MISSING_GPU | YES | NO | Gen2 geolocation, station extraction, 3dweather, and ML gridded products. |
| `XLONG` | MISSING_GPU | YES | NO | Gen2 geolocation, station extraction, 3dweather, and ML gridded products. |
| `XTIME` | MISSING_GPU | YES | NO | WRF elapsed-time coordinate used by thin gridded extraction. |

## Full Matrix

| Variable | CPU has | GPU writes | Dim agreement | Dtype agreement | Units agreement | Classification | Notes |
|---|---:|---:|---|---|---|---:|---|
| `ACCANHS` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACDEWC` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACDRIPR` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACDRIPS` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACECAN` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACEDIR` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACEFLXB` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACETLSM` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACETRAN` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACEVAC` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACEVB` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACEVC` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACEVG` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACFROC` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACFRZC` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACGHB` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACGHFLSM` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACGHV` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACGRDFLX` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACHFX` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACINTR` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACINTS` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACIRB` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACIRC` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACIRG` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACLHF` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACLHFLSM` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACLWDNB` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACLWDNBC` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACLWDNLSM` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACLWDNT` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACLWDNTC` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACLWUPB` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACLWUPBC` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACLWUPLSM` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACLWUPT` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACLWUPTC` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACMELTC` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACPAHB` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACPAHG` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACPAHLSM` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACPAHV` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACPONDING` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACQLAT` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACQRF` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACRAINLSM` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACRAINSNOW` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACRUNSB` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACRUNSF` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACSAGB` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACSAGV` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACSAV` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACSHB` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACSHC` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACSHFLSM` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACSHG` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACSNBOT` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACSNFRO` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACSNMELT` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACSNOM` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACSNOWLSM` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACSNSUB` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACSUBC` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACSWDNB` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACSWDNBC` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACSWDNLSM` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACSWDNT` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACSWDNTC` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACSWUPB` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACSWUPBC` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACSWUPLSM` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACSWUPT` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACSWUPTC` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACTHROR` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACTHROS` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ACTR` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ALBBCK` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ALBEDO` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ALBOLD` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `APAR` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `AREA2D` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `BATHYMETRY_FLAG` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `BGAP` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `C1F` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `C1H` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `C2F` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `C2H` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `C3F` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `C3H` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `C4F` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `C4H` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `CANICE` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `CANLIQ` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `CANWAT` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `CF1` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `CF2` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `CF3` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `CFN` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `CFN1` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `CH` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `CHB` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `CHB2` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `CHLEAF` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `CHSTAR` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `CHUC` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `CHV` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `CHV2` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `CLAT` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `CLDFRA` | YES | NO | N/A | N/A | N/A | MISSING_GPU | Cloud products; thin extractor derives TCC/CLDLOW/CLDMID/CLDHIGH. Downstream: Cloud products; thin extractor derives TCC/CLDLOW/CLDMID/CLDHIGH. |
| `CM` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `CON` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `COSALPHA` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `COSZEN` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `CROPCAT` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `DN` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `DNW` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `DTAUX3D` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `DTAUY3D` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `DUSFCG` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `DVSFCG` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `DX2D` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `DZS` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `E` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `EAH` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ECAN` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `EDIR` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `EL_PBL` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `EMISS` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ETRAN` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `EVB` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `EVC` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `EVG` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `F` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `FASTCP` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `FIRA` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `FNM` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `FNP` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `FORCPLSM` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `FORCQLSM` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `FORCTLSM` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `FORCWLSM` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `FORCZLSM` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `FSA` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `FVEG` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `FWET` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `GDD` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `GHB` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `GHV` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `GLW` | YES | NO | N/A | N/A | N/A | MISSING_GPU | Radiation/cloud feature and point-shadow products. Downstream: Radiation/cloud feature and point-shadow products. |
| `GOT_VAR_SSO` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `GPP` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `GRAIN` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `GRAUPELNC` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `GRDFLX` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `HAILNC` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `HFX` | YES | NO | N/A | N/A | N/A | MISSING_GPU | Surface heat-flux diagnostics. Downstream: Surface heat-flux diagnostics. |
| `HFX_FORCE` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `HFX_FORCE_TEND` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `HFX_KIN` | NO | YES | N/A | N/A | N/A | EXTRA_GPU | GPU-only kinematic heat-flux diagnostic; WRF HFX is W m-2. |
| `HGT` | YES | NO | N/A | N/A | N/A | MISSING_GPU | Terrain/georef audits and pressure-derived product proxies. Downstream: Terrain/georef audits and pressure-derived product proxies. |
| `IRB` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `IRC` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `IRELOSS` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `IRFIVOL` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `IRG` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `IRMIVOL` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `IRNUMFI` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `IRNUMMI` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `IRNUMSI` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `IRRSPLH` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `IRSIVOL` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ISEEDARRAY_SPP_CONV` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ISEEDARRAY_SPP_LSM` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ISEEDARRAY_SPP_PBL` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ISEEDARR_RAND_PERTURB` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ISEEDARR_SKEBS` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ISEEDARR_SPPT` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ISLTYP` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ISNOW` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ITIMESTEP` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `IVGTYP` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `LAI` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `LAKEMASK` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `LANDMASK` | YES | NO | N/A | N/A | N/A | MISSING_GPU | Surface/geography and cloud/terrain diagnostics. Downstream: Surface/geography and cloud/terrain diagnostics. |
| `LFMASS` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `LH` | YES | NO | N/A | N/A | N/A | MISSING_GPU | Latent heat-flux diagnostics. Downstream: Latent heat-flux diagnostics. |
| `LH_FORCE` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `LH_FORCE_TEND` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `LU_INDEX` | YES | NO | N/A | N/A | N/A | MISSING_GPU | Surface/geography diagnostics. Downstream: Surface/geography diagnostics. |
| `LWDNB` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `LWDNBC` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `LWDNT` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `LWDNTC` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `LWUPB` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `LWUPBC` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `LWUPT` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `LWUPTC` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `MAPFAC_M` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `MAPFAC_MX` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `MAPFAC_MY` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `MAPFAC_U` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `MAPFAC_UX` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `MAPFAC_UY` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `MAPFAC_V` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `MAPFAC_VX` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `MAPFAC_VY` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `MAXMF` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `MAXWIDTH` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `MAX_MSFTX` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `MAX_MSFTY` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `MF_VX_INV` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `MU` | YES | YES | NO: GPU omits singleton Time dimension | YES | YES | DEVIATION_DOCUMENTED | Current State.mu is aligned with mu_total; WRF MU is perturbation dry-column mass and MUB is omitted. |
| `MUB` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `NEE` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `NEST_POS` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `NOAHRES` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `NPP` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `O3_GFS_DU` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `OA1` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `OA2` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `OA3` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `OA4` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `OL1` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `OL2` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `OL3` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `OL4` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `OLR` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `P` | YES | YES | NO: GPU omits singleton Time dimension | YES | YES | DEVIATION_DOCUMENTED | Current State.p is aligned with p_total; WRF P is perturbation pressure and PB is omitted. |
| `P00` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `PB` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `PBLH` | YES | NO | N/A | N/A | N/A | MISSING_GPU | PBL diagnostics and point-shadow products. Downstream: PBL diagnostics and point-shadow products. |
| `PC` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `PCB` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `PGS` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `PH` | YES | YES | NO: GPU omits singleton Time dimension | YES | YES | DEVIATION_DOCUMENTED | Current State.ph is aligned with ph_total; WRF PH is perturbation geopotential and PHB is omitted. |
| `PHB` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `PSFC` | YES | NO | N/A | N/A | N/A | MISSING_GPU | Surface pressure, QA, and MSLP proxy input. Downstream: Surface pressure, QA, and MSLP proxy input. |
| `PSN` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `P_HYD` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `P_STRAT` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `P_TOP` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `Q2` | YES | YES | NO: GPU omits singleton Time dimension | YES | YES | DEVIATION_DOCUMENTED | Surface diagnostic present; Time dimension and NetCDF attributes omitted. Downstream: Surface humidity and validation loader field. |
| `Q2B` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `Q2V` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `QCLOUD` | YES | NO | N/A | N/A | N/A | MISSING_GPU | Cloud-cap diagnostics and cloud feature products. Downstream: Cloud-cap diagnostics and cloud feature products. |
| `QFX` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `QFX_KIN` | NO | YES | N/A | N/A | N/A | EXTRA_GPU | GPU-only kinematic moisture-flux diagnostic; WRF QFX uses mass flux units. |
| `QGRAUP` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `QICE` | YES | NO | N/A | N/A | N/A | MISSING_GPU | Cloud-cap diagnostics and cloud feature products. Downstream: Cloud-cap diagnostics and cloud feature products. |
| `QIN` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `QKE` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `QNICE` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `QNRAIN` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `QRAIN` | YES | NO | N/A | N/A | N/A | MISSING_GPU | Cloud/precip diagnostics. Downstream: Cloud/precip diagnostics. |
| `QRAINXY` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `QRFS` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `QSLAT` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `QSNOW` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `QSNOWXY` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `QSPRINGS` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `QTDRAIN` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `QVAPOR` | YES | YES | NO: GPU omits singleton Time dimension | YES | YES | DEVIATION_DOCUMENTED | WRF-shaped water-vapor field; writer omits Time dimension and NetCDF metadata. |
| `RAINC` | YES | NO | N/A | N/A | N/A | MISSING_GPU | Convective accumulated precipitation for precip verification. Downstream: Convective accumulated precipitation for precip verification. |
| `RAINNC` | YES | NO | N/A | N/A | N/A | MISSING_GPU | Grid-scale accumulated precipitation for precip verification. Downstream: Grid-scale accumulated precipitation for precip verification. |
| `RAINSH` | YES | NO | N/A | N/A | N/A | MISSING_GPU | Shallow-convective accumulated precipitation for precip verification. Downstream: Shallow-convective accumulated precipitation for precip verification. |
| `RDN` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `RDNW` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `RDX` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `RDY` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `RECH` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `RESM` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `RSSHA` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `RSSUN` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `RTMASS` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `RUNSB` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `RUNSF` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `SAG` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `SAV` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `SAVE_TOPO_FROM_REAL` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `SEAICE` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `SFROFF` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `SH2O` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `SHB` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `SHC` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `SHDAVG` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `SHDMAX` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `SHDMIN` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `SHG` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `SINALPHA` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `SMCWTD` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `SMOIS` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `SNEQVO` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `SNICE` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `SNLIQ` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `SNOALB` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `SNOW` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `SNOWC` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `SNOWENERGY` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `SNOWH` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `SNOWNC` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `SOILENERGY` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `SR` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `SST` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `SSTSK` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `SST_INPUT` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `STBLCP` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `STMASS` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `SWDNB` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `SWDNBC` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `SWDNT` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `SWDNTC` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `SWDOWN` | YES | NO | N/A | N/A | N/A | MISSING_GPU | Solar/cloud feature and point-shadow products. Downstream: Solar/cloud feature and point-shadow products. |
| `SWNORM` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `SWUPB` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `SWUPBC` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `SWUPT` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `SWUPTC` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `T` | YES | YES | NO: GPU omits singleton Time dimension | YES | YES | DEVIATION_DOCUMENTED | Writes state.theta - 300 K, matching WRF perturbation-theta convention; Time dimension omitted. |
| `T00` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `T2` | YES | YES | NO: GPU omits singleton Time dimension | YES | YES | DEVIATION_DOCUMENTED | Surface diagnostic present; Time dimension and NetCDF attributes omitted. Downstream: Binding M7 2m temperature and station verification field. |
| `T2B` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `T2V` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `TAH` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `TAUSS` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `TG` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `TGB` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `TGV` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `TH2` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `THIS_IS_AN_IDEAL_RUN` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `THM` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `TISO` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `TKE_PBL` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `TLP` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `TLP_STRAT` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `TMN` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `TR` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `TRAD` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `TSK` | YES | NO | N/A | N/A | N/A | MISSING_GPU | Skin-temperature diagnostics and surface-state QA. Downstream: Skin-temperature diagnostics and surface-state QA. |
| `TSK_FORCE` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `TSK_FORCE_TEND` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `TSLB` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `TSNO` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `TV` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `Times` | YES | NO | N/A | N/A | N/A | MISSING_GPU | WRF time coordinate used by raw wrfout and thin-NetCDF consumers. Downstream: WRF time coordinate used by raw wrfout and thin-NetCDF consumers. |
| `U` | YES | YES | NO: GPU omits singleton Time dimension | YES | YES | DEVIATION_DOCUMENTED | WRF-shaped staggered U array; writer omits Time dimension and NetCDF metadata. |
| `U10` | YES | YES | NO: GPU omits singleton Time dimension | YES | YES | DEVIATION_DOCUMENTED | Surface diagnostic present; Time dimension and NetCDF attributes omitted. Downstream: Binding M7 surface wind and station verification field. |
| `UDROFF` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `UST` | YES | YES | NO: GPU omits singleton Time dimension | YES | YES | DEVIATION_DOCUMENTED | Surface friction velocity present; Time dimension and NetCDF attributes omitted. Downstream: Surface-layer diagnostics and point-shadow products. |
| `V` | YES | YES | NO: GPU omits singleton Time dimension | YES | YES | DEVIATION_DOCUMENTED | WRF-shaped staggered V array; writer omits Time dimension and NetCDF metadata. |
| `V10` | YES | YES | NO: GPU omits singleton Time dimension | YES | YES | DEVIATION_DOCUMENTED | Surface diagnostic present; Time dimension and NetCDF attributes omitted. Downstream: Binding M7 surface wind and station verification field. |
| `VAR` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `VAR_SSO` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `VEGFRA` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `W` | YES | YES | NO: GPU omits singleton Time dimension | YES | YES | DEVIATION_DOCUMENTED | WRF-shaped staggered W array; writer omits Time dimension and NetCDF metadata. |
| `WA` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `WATER_DEPTH` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `WGAP` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `WOOD` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `WSLAKE` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `WT` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `XICEM` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `XLAND` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `XLAT` | YES | NO | N/A | N/A | N/A | MISSING_GPU | Gen2 geolocation, station extraction, 3dweather, and ML gridded products. Downstream: Gen2 geolocation, station extraction, 3dweather, and ML gridded products. |
| `XLAT_U` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `XLAT_V` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `XLONG` | YES | NO | N/A | N/A | N/A | MISSING_GPU | Gen2 geolocation, station extraction, 3dweather, and ML gridded products. Downstream: Gen2 geolocation, station extraction, 3dweather, and ML gridded products. |
| `XLONG_U` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `XLONG_V` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `XSAI` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `XTIME` | YES | NO | N/A | N/A | N/A | MISSING_GPU | WRF elapsed-time coordinate used by thin gridded extraction. Downstream: WRF elapsed-time coordinate used by thin gridded extraction. |
| `ZETATOP` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ZNU` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ZNW` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ZS` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ZSNSO` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ZTOP_PLUME` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `ZWT` | YES | NO | N/A | N/A | N/A | MISSING_GPU | CPU WRF variable not emitted by compact GPU writer. |
| `container_note` | NO | YES | N/A | N/A | N/A | EXTRA_GPU | GPU-only string metadata documenting the NPZ container. |
| `lead_hours` | NO | YES | N/A | N/A | N/A | EXTRA_GPU | GPU-only scalar metadata replacing WRF Times/XTIME. |
| `mass_shape` | NO | YES | N/A | N/A | N/A | EXTRA_GPU | GPU-only shape metadata. |
| `run_start_label` | NO | YES | N/A | N/A | N/A | EXTRA_GPU | GPU-only string metadata replacing WRF Times/XTIME. |
| `wrf_staggered_extent` | NO | YES | N/A | N/A | N/A | EXTRA_GPU | GPU-only shape metadata. |
