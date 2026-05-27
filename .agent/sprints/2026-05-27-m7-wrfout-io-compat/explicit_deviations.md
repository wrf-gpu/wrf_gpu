# Explicit wrfout Schema Deviations

Generated UTC: 2026-05-27T00:15:01+00:00

This document enumerates intentional schema differences found by the M7 audit. The complete row-level matrix is `compat_matrix.md`.

## Global Deviations

| Difference | Why | Downstream consumers care? | Action required |
|---|---|---|---|
| GPU writer emits `.npz` via `numpy.savez`, not WRF NetCDF4 `wrfout_d02_YYYY-MM-DD_HH:MM:SS`. | Compact proof container from the current forecast driver. | Yes; Gen2 raw-wrfout readers expect NetCDF variables, dimensions, attrs, and WRF filenames. | Re-implement NetCDF wrfout writer or provide a tested adapter before claiming drop-in compatibility. |
| GPU arrays omit the singleton `Time` dimension and named NetCDF dimensions. | Simplified direct array serialization. | Yes; xarray/netCDF4 consumers index `Time=0` and use named dimensions. | Add WRF-style dimensions or adapter. |
| GPU output omits WRF global attributes and per-variable attrs such as `units`, `description`, `MemoryOrder`, and `stagger`. | Simplified proof artifact. | Yes for auditability and some metadata-driven consumers. | Populate WRF-compatible attrs in NetCDF output. |
| GPU output stores `lead_hours` and `run_start_label` instead of `Times`/`XTIME`. | Simplified metadata. | Yes; Gen2 thin extraction and raw readers use WRF time coordinates. | Write `Times` and `XTIME`. |

## GPU-Written WRF Variables With Documented Deviations

| Variable | What is different | Why | Downstream consumers care? | Action required |
|---|---|---|---|---|
| `MU` | Current State.mu is aligned with mu_total; WRF MU is perturbation dry-column mass and MUB is omitted. | compact proof writer | maybe | write WRF perturbation variable and companion base-state variable, or rename total-state output |
| `P` | Current State.p is aligned with p_total; WRF P is perturbation pressure and PB is omitted. | compact proof writer | maybe | write WRF perturbation variable and companion base-state variable, or rename total-state output |
| `PH` | Current State.ph is aligned with ph_total; WRF PH is perturbation geopotential and PHB is omitted. | compact proof writer | maybe | write WRF perturbation variable and companion base-state variable, or rename total-state output |
| `Q2` | Surface diagnostic present; Time dimension and NetCDF attributes omitted. | compact proof writer | yes | document only for proof artifacts; re-implement NetCDF path for M7 drop-in use |
| `QVAPOR` | WRF-shaped water-vapor field; writer omits Time dimension and NetCDF metadata. | compact proof writer | maybe | document only for proof artifacts; re-implement NetCDF path for M7 drop-in use |
| `T` | Writes state.theta - 300 K, matching WRF perturbation-theta convention; Time dimension omitted. | compact proof writer | maybe | document only for proof artifacts; re-implement NetCDF path for M7 drop-in use |
| `T2` | Surface diagnostic present; Time dimension and NetCDF attributes omitted. | compact proof writer | yes | document only for proof artifacts; re-implement NetCDF path for M7 drop-in use |
| `U` | WRF-shaped staggered U array; writer omits Time dimension and NetCDF metadata. | compact proof writer | maybe | document only for proof artifacts; re-implement NetCDF path for M7 drop-in use |
| `U10` | Surface diagnostic present; Time dimension and NetCDF attributes omitted. | compact proof writer | yes | document only for proof artifacts; re-implement NetCDF path for M7 drop-in use |
| `UST` | Surface friction velocity present; Time dimension and NetCDF attributes omitted. | compact proof writer | yes | document only for proof artifacts; re-implement NetCDF path for M7 drop-in use |
| `V` | WRF-shaped staggered V array; writer omits Time dimension and NetCDF metadata. | compact proof writer | maybe | document only for proof artifacts; re-implement NetCDF path for M7 drop-in use |
| `V10` | Surface diagnostic present; Time dimension and NetCDF attributes omitted. | compact proof writer | yes | document only for proof artifacts; re-implement NetCDF path for M7 drop-in use |
| `W` | WRF-shaped staggered W array; writer omits Time dimension and NetCDF metadata. | compact proof writer | maybe | document only for proof artifacts; re-implement NetCDF path for M7 drop-in use |

## GPU-Only Payload Keys

| Key | Why | Downstream consumers care? | Action required |
|---|---|---|---|
| `HFX_KIN` | GPU-only kinematic heat-flux diagnostic; WRF HFX is W m-2. | No direct WRF consumer. | Keep only in proof artifacts; omit or map in NetCDF wrfout. |
| `QFX_KIN` | GPU-only kinematic moisture-flux diagnostic; WRF QFX uses mass flux units. | No direct WRF consumer. | Keep only in proof artifacts; omit or map in NetCDF wrfout. |
| `container_note` | GPU-only string metadata documenting the NPZ container. | No direct WRF consumer. | Keep only in proof artifacts; omit or map in NetCDF wrfout. |
| `lead_hours` | GPU-only scalar metadata replacing WRF Times/XTIME. | No direct WRF consumer. | Keep only in proof artifacts; omit or map in NetCDF wrfout. |
| `mass_shape` | GPU-only shape metadata. | No direct WRF consumer. | Keep only in proof artifacts; omit or map in NetCDF wrfout. |
| `run_start_label` | GPU-only string metadata replacing WRF Times/XTIME. | No direct WRF consumer. | Keep only in proof artifacts; omit or map in NetCDF wrfout. |
| `wrf_staggered_extent` | GPU-only shape metadata. | No direct WRF consumer. | Keep only in proof artifacts; omit or map in NetCDF wrfout. |

## Downstream-Critical CPU Variables Missing From GPU Output

| Variable | Why consumers care | Action required |
|---|---|---|
| `CLDFRA` | Cloud products; thin extractor derives TCC/CLDLOW/CLDMID/CLDHIGH. | Re-implement or explicitly replace in downstream adapter. |
| `GLW` | Radiation/cloud feature and point-shadow products. | Re-implement or explicitly replace in downstream adapter. |
| `HFX` | Surface heat-flux diagnostics. | Re-implement or explicitly replace in downstream adapter. |
| `HGT` | Terrain/georef audits and pressure-derived product proxies. | Re-implement or explicitly replace in downstream adapter. |
| `LANDMASK` | Surface/geography and cloud/terrain diagnostics. | Re-implement or explicitly replace in downstream adapter. |
| `LH` | Latent heat-flux diagnostics. | Re-implement or explicitly replace in downstream adapter. |
| `LU_INDEX` | Surface/geography diagnostics. | Re-implement or explicitly replace in downstream adapter. |
| `PBLH` | PBL diagnostics and point-shadow products. | Re-implement or explicitly replace in downstream adapter. |
| `PSFC` | Surface pressure, QA, and MSLP proxy input. | Re-implement or explicitly replace in downstream adapter. |
| `QCLOUD` | Cloud-cap diagnostics and cloud feature products. | Re-implement or explicitly replace in downstream adapter. |
| `QICE` | Cloud-cap diagnostics and cloud feature products. | Re-implement or explicitly replace in downstream adapter. |
| `QRAIN` | Cloud/precip diagnostics. | Re-implement or explicitly replace in downstream adapter. |
| `RAINC` | Convective accumulated precipitation for precip verification. | Re-implement or explicitly replace in downstream adapter. |
| `RAINNC` | Grid-scale accumulated precipitation for precip verification. | Re-implement or explicitly replace in downstream adapter. |
| `RAINSH` | Shallow-convective accumulated precipitation for precip verification. | Re-implement or explicitly replace in downstream adapter. |
| `SWDOWN` | Solar/cloud feature and point-shadow products. | Re-implement or explicitly replace in downstream adapter. |
| `TSK` | Skin-temperature diagnostics and surface-state QA. | Re-implement or explicitly replace in downstream adapter. |
| `Times` | WRF time coordinate used by raw wrfout and thin-NetCDF consumers. | Re-implement or explicitly replace in downstream adapter. |
| `XLAT` | Gen2 geolocation, station extraction, 3dweather, and ML gridded products. | Re-implement or explicitly replace in downstream adapter. |
| `XLONG` | Gen2 geolocation, station extraction, 3dweather, and ML gridded products. | Re-implement or explicitly replace in downstream adapter. |
| `XTIME` | WRF elapsed-time coordinate used by thin gridded extraction. | Re-implement or explicitly replace in downstream adapter. |

## Complete CPU Variable Omission List

The compact GPU writer omits 362 CPU WRF variables from the selected reference file:

`ACCANHS`, `ACDEWC`, `ACDRIPR`, `ACDRIPS`, `ACECAN`, `ACEDIR`, `ACEFLXB`, `ACETLSM`, `ACETRAN`,
`ACEVAC`, `ACEVB`, `ACEVC`, `ACEVG`, `ACFROC`, `ACFRZC`, `ACGHB`, `ACGHFLSM`, `ACGHV`, `ACGRDFLX`,
`ACHFX`, `ACINTR`, `ACINTS`, `ACIRB`, `ACIRC`, `ACIRG`, `ACLHF`, `ACLHFLSM`, `ACLWDNB`, `ACLWDNBC`,
`ACLWDNLSM`, `ACLWDNT`, `ACLWDNTC`, `ACLWUPB`, `ACLWUPBC`, `ACLWUPLSM`, `ACLWUPT`, `ACLWUPTC`,
`ACMELTC`, `ACPAHB`, `ACPAHG`, `ACPAHLSM`, `ACPAHV`, `ACPONDING`, `ACQLAT`, `ACQRF`, `ACRAINLSM`,
`ACRAINSNOW`, `ACRUNSB`, `ACRUNSF`, `ACSAGB`, `ACSAGV`, `ACSAV`, `ACSHB`, `ACSHC`, `ACSHFLSM`,
`ACSHG`, `ACSNBOT`, `ACSNFRO`, `ACSNMELT`, `ACSNOM`, `ACSNOWLSM`, `ACSNSUB`, `ACSUBC`, `ACSWDNB`,
`ACSWDNBC`, `ACSWDNLSM`, `ACSWDNT`, `ACSWDNTC`, `ACSWUPB`, `ACSWUPBC`, `ACSWUPLSM`, `ACSWUPT`,
`ACSWUPTC`, `ACTHROR`, `ACTHROS`, `ACTR`, `ALBBCK`, `ALBEDO`, `ALBOLD`, `APAR`, `AREA2D`,
`BATHYMETRY_FLAG`, `BGAP`, `C1F`, `C1H`, `C2F`, `C2H`, `C3F`, `C3H`, `C4F`, `C4H`, `CANICE`,
`CANLIQ`, `CANWAT`, `CF1`, `CF2`, `CF3`, `CFN`, `CFN1`, `CH`, `CHB`, `CHB2`, `CHLEAF`, `CHSTAR`,
`CHUC`, `CHV`, `CHV2`, `CLAT`, `CLDFRA`, `CM`, `CON`, `COSALPHA`, `COSZEN`, `CROPCAT`, `DN`, `DNW`,
`DTAUX3D`, `DTAUY3D`, `DUSFCG`, `DVSFCG`, `DX2D`, `DZS`, `E`, `EAH`, `ECAN`, `EDIR`, `EL_PBL`,
`EMISS`, `ETRAN`, `EVB`, `EVC`, `EVG`, `F`, `FASTCP`, `FIRA`, `FNM`, `FNP`, `FORCPLSM`, `FORCQLSM`,
`FORCTLSM`, `FORCWLSM`, `FORCZLSM`, `FSA`, `FVEG`, `FWET`, `GDD`, `GHB`, `GHV`, `GLW`,
`GOT_VAR_SSO`, `GPP`, `GRAIN`, `GRAUPELNC`, `GRDFLX`, `HAILNC`, `HFX`, `HFX_FORCE`,
`HFX_FORCE_TEND`, `HGT`, `IRB`, `IRC`, `IRELOSS`, `IRFIVOL`, `IRG`, `IRMIVOL`, `IRNUMFI`, `IRNUMMI`,
`IRNUMSI`, `IRRSPLH`, `IRSIVOL`, `ISEEDARRAY_SPP_CONV`, `ISEEDARRAY_SPP_LSM`, `ISEEDARRAY_SPP_PBL`,
`ISEEDARR_RAND_PERTURB`, `ISEEDARR_SKEBS`, `ISEEDARR_SPPT`, `ISLTYP`, `ISNOW`, `ITIMESTEP`,
`IVGTYP`, `LAI`, `LAKEMASK`, `LANDMASK`, `LFMASS`, `LH`, `LH_FORCE`, `LH_FORCE_TEND`, `LU_INDEX`,
`LWDNB`, `LWDNBC`, `LWDNT`, `LWDNTC`, `LWUPB`, `LWUPBC`, `LWUPT`, `LWUPTC`, `MAPFAC_M`, `MAPFAC_MX`,
`MAPFAC_MY`, `MAPFAC_U`, `MAPFAC_UX`, `MAPFAC_UY`, `MAPFAC_V`, `MAPFAC_VX`, `MAPFAC_VY`, `MAXMF`,
`MAXWIDTH`, `MAX_MSFTX`, `MAX_MSFTY`, `MF_VX_INV`, `MUB`, `NEE`, `NEST_POS`, `NOAHRES`, `NPP`,
`O3_GFS_DU`, `OA1`, `OA2`, `OA3`, `OA4`, `OL1`, `OL2`, `OL3`, `OL4`, `OLR`, `P00`, `PB`, `PBLH`,
`PC`, `PCB`, `PGS`, `PHB`, `PSFC`, `PSN`, `P_HYD`, `P_STRAT`, `P_TOP`, `Q2B`, `Q2V`, `QCLOUD`,
`QFX`, `QGRAUP`, `QICE`, `QIN`, `QKE`, `QNICE`, `QNRAIN`, `QRAIN`, `QRAINXY`, `QRFS`, `QSLAT`,
`QSNOW`, `QSNOWXY`, `QSPRINGS`, `QTDRAIN`, `RAINC`, `RAINNC`, `RAINSH`, `RDN`, `RDNW`, `RDX`, `RDY`,
`RECH`, `RESM`, `RSSHA`, `RSSUN`, `RTMASS`, `RUNSB`, `RUNSF`, `SAG`, `SAV`, `SAVE_TOPO_FROM_REAL`,
`SEAICE`, `SFROFF`, `SH2O`, `SHB`, `SHC`, `SHDAVG`, `SHDMAX`, `SHDMIN`, `SHG`, `SINALPHA`, `SMCWTD`,
`SMOIS`, `SNEQVO`, `SNICE`, `SNLIQ`, `SNOALB`, `SNOW`, `SNOWC`, `SNOWENERGY`, `SNOWH`, `SNOWNC`,
`SOILENERGY`, `SR`, `SST`, `SSTSK`, `SST_INPUT`, `STBLCP`, `STMASS`, `SWDNB`, `SWDNBC`, `SWDNT`,
`SWDNTC`, `SWDOWN`, `SWNORM`, `SWUPB`, `SWUPBC`, `SWUPT`, `SWUPTC`, `T00`, `T2B`, `T2V`, `TAH`,
`TAUSS`, `TG`, `TGB`, `TGV`, `TH2`, `THIS_IS_AN_IDEAL_RUN`, `THM`, `TISO`, `TKE_PBL`, `TLP`,
`TLP_STRAT`, `TMN`, `TR`, `TRAD`, `TSK`, `TSK_FORCE`, `TSK_FORCE_TEND`, `TSLB`, `TSNO`, `TV`,
`Times`, `UDROFF`, `VAR`, `VAR_SSO`, `VEGFRA`, `WA`, `WATER_DEPTH`, `WGAP`, `WOOD`, `WSLAKE`, `WT`,
`XICEM`, `XLAND`, `XLAT`, `XLAT_U`, `XLAT_V`, `XLONG`, `XLONG_U`, `XLONG_V`, `XSAI`, `XTIME`,
`ZETATOP`, `ZNU`, `ZNW`, `ZS`, `ZSNSO`, `ZTOP_PLUME`, `ZWT`
