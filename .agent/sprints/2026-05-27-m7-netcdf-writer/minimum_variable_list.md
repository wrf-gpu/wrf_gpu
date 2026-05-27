# Minimum wrfout Variable List

Sprint: `2026-05-27-m7-netcdf-writer`
Reference audit inputs: `.agent/sprints/2026-05-27-m7-wrfout-io-compat/compat_matrix.md` and `explicit_deviations.md`.
Decision: emit 41 variables, not the full 375-variable Gen2 reference file. The set covers every downstream-critical variable from the audit plus the core WRF dynamic fields and staggered coordinate variables needed to keep the NetCDF schema coherent.

## Downstream-Critical Variables

| Variable | Consumer | Reasoning |
|---|---|---|
| `Times` | raw wrfout readers, thin-NetCDF extraction | WRF-standard timestamp coordinate used by existing readers. |
| `XTIME` | raw wrfout readers, thin-NetCDF extraction | Elapsed-minute coordinate used by Gen2 time slicing. |
| `XLAT` | Gen2 post-processing, station extraction, 3dweather, ML gridded products | Mass-grid latitude for geolocation and station interpolation. |
| `XLONG` | Gen2 post-processing, station extraction, 3dweather, ML gridded products | Mass-grid longitude for geolocation and station interpolation. |
| `HGT` | terrain/georef QA, pressure-derived product proxies | Terrain height is required for georef audits and surface diagnostics. |
| `LANDMASK` | surface/geography products, cloud/terrain diagnostics | Distinguishes land and water points for masks and QA. |
| `LU_INDEX` | surface/geography diagnostics | Land-use category used by surface product diagnostics. |
| `U10` | AEMET station verification, surface wind products | Binding 10 m wind field. |
| `V10` | AEMET station verification, surface wind products | Binding 10 m wind field. |
| `T2` | AEMET station verification, surface temperature products | Binding 2 m temperature field. |
| `Q2` | surface humidity verification and loaders | 2 m humidity needed by station and surface QA paths. |
| `PSFC` | pressure QA and MSLP proxy input | Surface pressure is consumed by QA and pressure products. |
| `RAINC` | precipitation verification | Convective accumulated precipitation component. |
| `RAINNC` | precipitation verification | Grid-scale accumulated precipitation component. |
| `RAINSH` | precipitation verification | Shallow-convective accumulated precipitation component. |
| `SWDOWN` | solar/cloud products, point-shadow products | Downward shortwave radiation at surface. |
| `GLW` | radiation/cloud products, point-shadow products | Downward longwave radiation at surface. |
| `PBLH` | PBL diagnostics and point products | Boundary-layer height diagnostic. |
| `UST` | surface-layer diagnostics and point products | Friction velocity diagnostic. |
| `HFX` | heat-flux diagnostics | Upward sensible heat flux in W m-2. |
| `LH` | latent-heat diagnostics | Upward latent heat flux in W m-2. |
| `TSK` | skin-temperature products and surface-state QA | Surface skin temperature. |
| `CLDFRA` | cloud products, thin-NetCDF cloud levels | Cloud fraction used to derive total/low/mid/high cloud products. |
| `QCLOUD` | cloud-cap and cloud-feature products | Cloud water mixing ratio. |
| `QICE` | cloud-cap and cloud-feature products | Ice mixing ratio. |
| `QRAIN` | cloud and precipitation diagnostics | Rain water mixing ratio. |

## Core WRF Coherence Variables

| Variable | Consumer | Reasoning |
|---|---|---|
| `XLAT_U` | WRF-aware readers of staggered U | Coordinate variable referenced by WRF `U.coordinates`. |
| `XLONG_U` | WRF-aware readers of staggered U | Coordinate variable referenced by WRF `U.coordinates`. |
| `XLAT_V` | WRF-aware readers of staggered V | Coordinate variable referenced by WRF `V.coordinates`. |
| `XLONG_V` | WRF-aware readers of staggered V | Coordinate variable referenced by WRF `V.coordinates`. |
| `U` | dynamic-state diagnostics and WRF compatibility checks | Staggered x-wind field. |
| `V` | dynamic-state diagnostics and WRF compatibility checks | Staggered y-wind field. |
| `W` | dynamic-state diagnostics and WRF compatibility checks | Staggered vertical wind field. |
| `T` | dynamic-state diagnostics and WRF compatibility checks | WRF perturbation potential temperature, `theta - 300 K`. |
| `QVAPOR` | humidity diagnostics and compatibility checks | Water-vapor mixing ratio on mass grid. |
| `P` | pressure diagnostics and base/perturbation compatibility | WRF perturbation pressure. |
| `PB` | pressure diagnostics and base/perturbation compatibility | WRF base-state pressure companion to `P`. |
| `PH` | geopotential diagnostics and base/perturbation compatibility | WRF perturbation geopotential. |
| `PHB` | geopotential diagnostics and base/perturbation compatibility | WRF base-state geopotential companion to `PH`. |
| `MU` | dry-column mass diagnostics and base/perturbation compatibility | WRF perturbation dry air mass in column. |
| `MUB` | dry-column mass diagnostics and base/perturbation compatibility | WRF base-state dry air mass companion to `MU`. |

Out of scope for this sprint: the remaining non-consumed Gen2 reference variables, native `wrfinput`/`wrfbdy`/`wrfrst` production, and physical validation of diagnostics that the current GPU state does not yet compute.
