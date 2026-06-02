# GPT v0.3.0 S4 WPS Parity Oracle Harness

Date: 2026-06-02
Branch: `worker/gpt/v030-parity`
Role: GPT-5.5 xhigh oracle lane
Scope: `proofs/v030/parity/` and this review/spec only.

## Objective

Build the v0.3.0 WPS/metgrid parity oracle side before ingest implementation:

- Inventory the real WPS `met_em.d0*.nc` schema.
- Inventory current AIFS forcing and map it to `met_em`.
- Add a CPU-only parity scorer that compares native metgrid-equivalent output against real WPS `met_em`.
- Predeclare per-variable tolerances and prove the scorer on real-data self parity plus a perturbed-copy negative test.

Note: the requested worktree command was followed exactly from local branch
`worker/opus/v030-native-init`, which currently points at `b301baa`. The sprint
plan file named in the dispatch is present in the main worktree at `370f948`,
not at that local branch tip, so it was read from
`/home/enric/src/wrf_gpu2/.agent/decisions/SPRINT-PLAN-V030-V040.md`.

## Proof Objects

- `proofs/v030/parity/metem_parity_scorer.py`
- `proofs/v030/parity/metem_self_parity.json`
- `proofs/v030/parity/metem_perturbed_parity.json`

## met_em Corpus Inventory

Oracle root:
`/mnt/data/canairy_meteo/runs/wps_cases/<case>/l3/`

Cases found: 13

1. `20260428_18z_72h`
2. `20260429_18z_72h`
3. `20260521_18z_72h`
4. `20260521_18z_l2rerun_72h`
5. `20260522_18z_72h`
6. `20260523_18z_72h`
7. `20260524_18z_72h`
8. `20260525_18z_72h`
9. `20260527_18z_72h`
10. `20260528_18z_72h`
11. `20260529_18z_72h`
12. `20260530_18z_72h`
13. `20260531_18z_72h`

Each case has 13 six-hourly `met_em` times for d01-d05. The v0.3.0 S4 scorer
defaults to d01-d03 per dispatch: 13 cases x 13 times x 3 domains = 507 files.
d04 and d05 are present and have the same 70-variable schema but are not in the
default gate.

### Dimensions

| domain | grid_id | parent | ratio | DX/DY | west_east | south_north | west_east_stag | south_north_stag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| d01 | 1 | 1 | 1 | 9000 m | 93 | 59 | 94 | 60 |
| d02 | 2 | 1 | 3 | 3000 m | 159 | 66 | 160 | 67 |
| d03 | 3 | 2 | 3 | 1000 m | 93 | 75 | 94 | 76 |

Shared dimensions:

- `Time = 1`, `DateStrLen = 19`
- `num_metgrid_levels = 14`
- `num_st_layers = 2`, `num_sm_layers = 2`
- `z-dimension0012 = 12`, `z-dimension0016 = 16`,
  `z-dimension0021 = 21`

Pressure/metgrid levels in these files:

`surface pressure`, `1000`, `925`, `850`, `700`, `600`, `500`, `400`, `300`,
`250`, `200`, `150`, `100`, `50` hPa.

The `PRES` level-0 field is spatially varying surface pressure. Levels 1-13 are
fixed pressure surfaces in Pa. `SPECHUMD` level 0 is the WPS sentinel `-1`.

Soil layer coordinate values in `SOIL_LAYERS`: `40`, `10` cm, matching
`010040` and `000010` layer products in this WPS output ordering.

### Projection and Key Global Attributes

Projection: Lambert conformal (`MAP_PROJ = 1`) with:

- `TRUELAT1 = 25.0`
- `TRUELAT2 = 30.0`
- `STAND_LON = -16.4`
- `MOAD_CEN_LAT = 28.300007`
- `POLE_LAT = 90.0`
- `POLE_LON = 0.0`
- `MMINLU = MODIFIED_IGBP_MODIS_NOAH`
- `NUM_LAND_CAT = 21`
- `ISWATER = 17`, `ISLAKE = 21`, `ISICE = 15`, `ISURBAN = 13`,
  `ISOILWATER = 14`
- Required WPS flags present: `FLAG_METGRID`, `FLAG_SOIL_LAYERS`,
  `FLAG_PSFC`, `FLAG_SM000010`, `FLAG_SM010040`, `FLAG_ST000010`,
  `FLAG_ST010040`, `FLAG_SLP`, `FLAG_SH`, `FLAG_SOILHGT`, `FLAG_MF_XY`,
  `FLAG_LAI12M`

Domain centers and WPS `corner_lats` / `corner_lons` from sample case
`20260521_18z_72h`, first valid time:

| domain | CEN_LAT | CEN_LON | corner_lats | corner_lons |
|---|---:|---:|---|---|
| d01 | 28.300007 | -16.399994 | 25.888184, 30.583416, 30.583416, 25.888184, 25.886818, 30.581997, 30.581997, 25.886818, 25.847694, 30.623844, 30.623844, 25.847694, 25.846329, 30.622417, 30.622417, 25.846329 | -20.542206, -20.724213, -12.075806, -12.257782, -20.587189, -20.771179, -12.028839, -12.212799, -20.540710, -20.725830, -12.074158, -12.259277, -20.585693, -20.772827, -12.027161, -12.214325 |
| d02 | 28.340271 | -16.123840 | 27.446522, 29.201641, 29.192059, 27.437096, 27.446278, 29.201389, 29.191772, 27.436794, 27.433025, 29.215126, 29.205563, 27.423592, 27.432781, 29.214893, 29.205254, 27.423298 | -18.530579, -18.565094, -13.678314, -13.721710, -18.545807, -18.580566, -13.662842, -13.706512, -18.530304, -18.565369, -13.677979, -13.722046, -18.545532, -18.580841, -13.662506, -13.706818 |
| d03 | 28.299953 | -16.522705 | 27.965679, 28.631832, 28.632652, 27.966484, 27.965645, 28.631821, 28.632637, 27.966469, 27.961178, 28.636333, 28.637154, 27.961990, 27.961151, 28.636314, 28.637146, 27.961983 | -16.991211, -16.994843, -16.051331, -16.053436, -16.996307, -16.999969, -16.046204, -16.048340, -16.991211, -16.994873, -16.051300, -16.053467, -16.996277, -17.000000, -16.046173, -16.048370 |

### Complete Variable Schema

All sampled d01-d03 files have 70 variables. Variable signatures are stable
across all 507 d01-d03 files. The only observed metadata-text variation is that
`HGT_M` and `LANDUSEF` carry richer `units`/`description` strings on d01 than on
d02/d03 in the WPS files; the parity scorer compares native metadata to the
matching oracle file, not to a normalized cross-domain string.

| variable | dims | stagger | units | source class |
|---|---|---|---|---|
| Times | Time x DateStrLen | - | - | time |
| PRES | Time x num_metgrid_levels x south_north x west_east | M | - | derived_from_aifs_plus_wps_levels |
| SOIL_LAYERS | Time x num_st_layers x south_north x west_east | M | - | wps_constant |
| SM | Time x num_sm_layers x south_north x west_east | M | - | aifs_gap |
| ST | Time x num_st_layers x south_north x west_east | M | - | aifs_gap |
| GHT | Time x num_metgrid_levels x south_north x west_east | M | m | partial_aifs_gap |
| SPECHUMD | Time x num_metgrid_levels x south_north x west_east | M | kg kg-1 | aifs_gap |
| ST010040 | Time x south_north x west_east | M | K | aifs_gap |
| ST000010 | Time x south_north x west_east | M | K | aifs_gap |
| SM010040 | Time x south_north x west_east | M | fraction | aifs_gap |
| SM000010 | Time x south_north x west_east | M | fraction | aifs_gap |
| SOILHGT | Time x south_north x west_east | M | m | aifs_gap |
| SKINTEMP | Time x south_north x west_east | M | K | aifs_gap |
| LANDSEA | Time x south_north x west_east | M | 0/1 Flag | static_geog |
| PSFC | Time x south_north x west_east | M | Pa | aifs_direct |
| DEWPT | Time x south_north x west_east | M | K | aifs_direct |
| VV | Time x num_metgrid_levels x south_north_stag x west_east | V | m s-1 | aifs_gap |
| UU | Time x num_metgrid_levels x south_north x west_east_stag | U | m s-1 | aifs_gap |
| TT | Time x num_metgrid_levels x south_north x west_east | M | K | partial_aifs_gap |
| PMSL | Time x south_north x west_east | M | Pa | aifs_direct |
| OL4 | Time x south_north x west_east | M | whoknows | static_geog |
| OL3 | Time x south_north x west_east | M | whoknows | static_geog |
| OL2 | Time x south_north x west_east | M | whoknows | static_geog |
| OL1 | Time x south_north x west_east | M | whoknows | static_geog |
| OA4 | Time x south_north x west_east | M | whoknows | static_geog |
| OA3 | Time x south_north x west_east | M | whoknows | static_geog |
| OA2 | Time x south_north x west_east | M | whoknows | static_geog |
| OA1 | Time x south_north x west_east | M | whoknows | static_geog |
| VAR | Time x south_north x west_east | M | whoknows | static_geog |
| CON | Time x south_north x west_east | M | whoknows | static_geog |
| SNOALB | Time x south_north x west_east | M | percent | static_geog |
| LAI12M | Time x z-dimension0012 x south_north x west_east | M | m^2/m^2 | static_geog |
| GREENFRAC | Time x z-dimension0012 x south_north x west_east | M | fraction | static_geog |
| ALBEDO12M | Time x z-dimension0012 x south_north x west_east | M | percent | static_geog |
| SCB_DOM | Time x south_north x west_east | M | category | static_geog |
| SOILCBOT | Time x z-dimension0016 x south_north x west_east | M | category | static_geog |
| SCT_DOM | Time x south_north x west_east | M | category | static_geog |
| SOILCTOP | Time x z-dimension0016 x south_north x west_east | M | category | static_geog |
| SOILTEMP | Time x south_north x west_east | M | Kelvin | static_geog |
| HGT_M | Time x south_north x west_east | M | meters MSL | static_geog |
| LU_INDEX | Time x south_north x west_east | M | category | static_geog |
| LANDUSEF | Time x z-dimension0021 x south_north x west_east | M | category | static_geog |
| COSALPHA_V | Time x south_north_stag x west_east | V | none | projection_static |
| SINALPHA_V | Time x south_north_stag x west_east | V | none | projection_static |
| COSALPHA_U | Time x south_north x west_east_stag | U | none | projection_static |
| SINALPHA_U | Time x south_north x west_east_stag | U | none | projection_static |
| XLONG_C | Time x south_north_stag x west_east_stag | CORNER | degrees longitude | projection_static |
| XLAT_C | Time x south_north_stag x west_east_stag | CORNER | degrees latitude | projection_static |
| LANDMASK | Time x south_north x west_east | M | none | static_geog |
| COSALPHA | Time x south_north x west_east | M | none | projection_static |
| SINALPHA | Time x south_north x west_east | M | none | projection_static |
| F | Time x south_north x west_east | M | - | projection_static |
| E | Time x south_north x west_east | M | - | projection_static |
| MAPFAC_UY | Time x south_north x west_east_stag | U | none | projection_static |
| MAPFAC_VY | Time x south_north_stag x west_east | V | none | projection_static |
| MAPFAC_MY | Time x south_north x west_east | M | none | projection_static |
| MAPFAC_UX | Time x south_north x west_east_stag | U | none | projection_static |
| MAPFAC_VX | Time x south_north_stag x west_east | V | none | projection_static |
| MAPFAC_MX | Time x south_north x west_east | M | none | projection_static |
| MAPFAC_U | Time x south_north x west_east_stag | U | none | projection_static |
| MAPFAC_V | Time x south_north_stag x west_east | V | none | projection_static |
| MAPFAC_M | Time x south_north x west_east | M | none | projection_static |
| CLONG | Time x south_north x west_east | M | degrees longitude | projection_static |
| CLAT | Time x south_north x west_east | M | degrees latitude | projection_static |
| XLONG_U | Time x south_north x west_east_stag | U | degrees longitude | projection_static |
| XLAT_U | Time x south_north x west_east_stag | U | degrees latitude | projection_static |
| XLONG_V | Time x south_north_stag x west_east | V | degrees longitude | projection_static |
| XLAT_V | Time x south_north_stag x west_east | V | degrees latitude | projection_static |
| XLONG_M | Time x south_north x west_east | M | degrees longitude | projection_static |
| XLAT_M | Time x south_north x west_east | M | degrees latitude | projection_static |

## AIFS Inventory

Base AIFS root:
`/mnt/data/canairy_meteo/data/aifs_single/aifs_single_YYYYMM.nc`

Sample `aifs_single_202605.nc`:

- Dims: `init_time = 124`, `lead_time = 61`, `latitude = 13`,
  `longitude = 23`
- Grid: 0.25 degree Canary bbox, lat 26.5..29.5, lon -18.5..-13.0
- Time: 6-hourly issues and 0..360 h lead times
- Fields: `wind_u_10m`, `wind_v_10m`, `wind_u_100m`, `wind_v_100m`,
  `temperature_2m`, `total_cloud_cover_atmosphere`,
  `precipitation_surface`, `pressure_reduced_to_mean_sea_level`

Expanded sidecar root:
`/mnt/data/canairy_meteo/data/aifs_single_expanded_fields_v1/aifs_single_expanded_YYYYMM.nc`

Sample expanded fields:

- `dew_point_temperature_2m`
- `downward_long_wave_radiation_flux_surface`
- `downward_short_wave_radiation_flux_surface`
- `pressure_surface`
- `temperature_850hpa`
- `temperature_925hpa`
- `geopotential_height_500hpa`
- `geopotential_height_850hpa`
- `geopotential_height_925hpa`

Coverage warning:

- `aifs_single_expanded_202604.nc`: 120 issue times, 2026-04-01 00Z through
  2026-04-30 18Z.
- `aifs_single_expanded_202605.nc`: only 4 issue times, 2026-05-01 00Z through
  2026-05-01 18Z.

Most v0.3 oracle cases are 2026-05-21 and later, so the expanded fields needed
for `PSFC`, `DEWPT`, and partial pressure-level `TT`/`GHT` are not materially
available for those cases in the current sidecar inventory.

## AIFS->met_em Variable Map and Gaps

| met_em variable(s) | source | derivation/interp | gap status |
|---|---|---|---|
| `Times` | case valid time | format as WRF DateStrLen=19 | OK |
| `PMSL` | base `pressure_reduced_to_mean_sea_level` | direct Pa, horizontal interpolation to mass grid | OK |
| `PSFC` | expanded `pressure_surface` | direct Pa, horizontal interpolation to mass grid | GAP for May21+ expanded coverage |
| `DEWPT` | expanded `dew_point_temperature_2m` | Celsius to Kelvin, horizontal interpolation | GAP for May21+ expanded coverage |
| `PRES` | expanded `pressure_surface` plus WPS fixed pressure list | level 0 surface pressure; levels 1-13 fixed pressure surfaces | partial; depends on missing May expanded `pressure_surface` |
| `TT` | expanded `temperature_925hpa`, `temperature_850hpa` | Celsius to Kelvin, horizontal interpolation; other pressure levels unavailable | GAP: no 1000/700/600/500/400/300/250/200/150/100/50 hPa temperature |
| `GHT` | expanded `geopotential_height_925hpa`, `_850hpa`, `_500hpa` | meters, horizontal interpolation; other pressure levels unavailable | GAP: no surface/1000/700/600/400/300/250/200/150/100/50 hPa height |
| `UU`, `VV` | none suitable in current AIFS archive | 10m/100m winds are not pressure-level winds and cannot fill 14 metgrid levels | GAP: pressure-level U/V winds absent |
| `SPECHUMD` | none suitable in current AIFS archive | 2m dewpoint can derive near-surface humidity only; WPS needs pressure levels | GAP: pressure-level humidity absent |
| `SKINTEMP` | none in current AIFS archive | requires skin/SST source or approved derivation; 2m temperature is not equivalent | GAP |
| `SM`, `SM000010`, `SM010040` | none in current AIFS archive | requires soil moisture source/fallback for 0-10 and 10-40 cm | GAP |
| `ST`, `ST000010`, `ST010040` | none in current AIFS archive | requires soil temperature source/fallback for 0-10 and 10-40 cm | GAP |
| `SOILHGT` | not in current AIFS archive | WPS source-model surface height; fallback to `HGT_M` would be a documented approximation | GAP / policy decision needed |
| `SOIL_LAYERS` | WPS/metgrid coordinate | reproduce two-layer coordinate values `40`, `10` cm | OK |
| `LANDSEA`, `LANDMASK`, `LU_INDEX`, `LANDUSEF`, `SCT_DOM`, `SCB_DOM`, `SOILCTOP`, `SOILCBOT`, `SNOALB`, `LAI12M`, `GREENFRAC`, `ALBEDO12M`, `SOILTEMP`, `HGT_M`, `OL*`, `OA*`, `VAR`, `CON` | `geo_em.d0*.nc` / geogrid | copy into the native artifact with WPS dimensions/metadata | OK static-geog, not AIFS |
| `XLAT_*`, `XLONG_*`, `CLAT`, `CLONG`, `MAPFAC_*`, `COSALPHA*`, `SINALPHA*`, `F`, `E` | `geo_em.d0*.nc` / WPS projection metadata | copy from geo_em or recompute bit-faithfully, respecting M/U/V/corner staggering | OK projection-static, not AIFS |

Bottom line: the current AIFS archive is not sufficient for a WRF-faithful
metgrid-equivalent ingest without adding fields or approving explicit fallback
policies. The largest true gaps are pressure-level winds, pressure-level
humidity, most pressure-level temperature/height fields, soil fields, skin/SST,
and May21+ expanded-field coverage.

## Predeclared Tolerances

Tolerances are encoded in `metem_parity_scorer.py` and emitted in each proof
JSON under `variable_policies` and `variable_results[*].tolerance`.

| variable(s) | RMSE | abs bias | max abs | rationale |
|---|---:|---:|---:|---|
| `Times`, categorical masks/classes, `SOIL_LAYERS` | 0 | 0 | 0 | exact WRF metadata/category match |
| `TT`, `DEWPT`, `SKINTEMP` | 0.05 K for TT; 0.1 K for surface temp fields | 0.01/0.02 K | 0.5/1.0 K | tight interpolated temperature parity |
| `UU`, `VV` | 0.05 m s-1 | 0.01 m s-1 | 0.5 m s-1 | tight staggered wind parity once source exists |
| `GHT` | 1.0 m | 0.2 m | 10.0 m | pressure-level height interpolation tolerance |
| `SPECHUMD` | 1e-5 kg kg-1 | 2e-6 kg kg-1 | 1e-4 kg kg-1 | humidity profile tolerance plus exact sentinel handling |
| `PRES` | 0.2 Pa | 0.05 Pa | 2 Pa | fixed pressure levels plus interpolated surface level |
| `PSFC`, `PMSL` | 5 Pa | 1 Pa | 50 Pa | surface pressure/MSLP interpolation tolerance |
| `SM*` | 0.002 fraction | 0.0005 fraction | 0.02 fraction | soil moisture fallback/source tolerance |
| `ST*` | 0.25 K | 0.05 K | 2.0 K | soil temperature fallback/source tolerance |
| `HGT_M`, `SOILHGT` | 0.001 m | 0.0001 m | 0.01 m | near-exact terrain/static-height parity |
| static monthly/geog floats | 1e-6 | 1e-7 | 1e-5 | copy/recompute should be near-exact |
| coordinates | 1e-6 deg | 1e-7 deg | 2e-5 deg | WPS projection-coordinate parity |
| map factors, rotation, Coriolis | 1e-7 | 1e-8 | 1e-6 | near-exact projection fields |

The scorer also fails on missing native files, missing variables, shape
mismatches, variable metadata mismatches, key global-attribute mismatches, and
oracle/native NaN-mask mismatches, independent of numeric tolerances.

## Harness Usage

Native artifact layout supported:

- Preferred full root:
  `<native-root>/<case>/l3/met_em.d0N.YYYY-MM-DD_HH:MM:SS.nc`
- Also accepted for single-case/scratch roots:
  `<native-root>/<case>/met_em...`, `<native-root>/l3/met_em...`, or
  `<native-root>/met_em...`

Full 13-case d01-d03 self sanity command:

```bash
PYTHONPATH=src taskset -c 0-3 python proofs/v030/parity/metem_parity_scorer.py \
  --oracle-root /mnt/data/canairy_meteo/runs/wps_cases \
  --native-root /mnt/data/canairy_meteo/runs/wps_cases \
  --domains d01,d02,d03 \
  --output proofs/v030/parity/metem_self_parity.json
```

Result:

- status: `PASS`
- compared files: 507
- variable failures: none
- representative metrics: `TT`, `PRES`, `UU`, `XLAT_M` all RMSE/bias/max = 0

Perturbed-copy negative command:

```bash
STUB=$(mktemp -d /tmp/metem_stub.XXXXXX)
PYTHONPATH=src taskset -c 0-3 python proofs/v030/parity/metem_parity_scorer.py \
  --oracle-root /mnt/data/canairy_meteo/runs/wps_cases \
  --make-stub-root "$STUB" \
  --cases 20260521_18z_72h \
  --domains d03 \
  --perturb-variable TT \
  --perturb-delta 1.0 \
  --allow-failures \
  --output proofs/v030/parity/metem_perturbed_parity.json
```

Result:

- status: `FAIL` as expected
- compared files: 13
- variable failures: `TT` only
- `TT` metrics: RMSE `0.2773500981126146`, bias `0.07692307692307693`,
  max abs diff `1.0`

## Commands Run

- `git worktree add -b worker/gpt/v030-parity /home/enric/src/wrf_gpu2/.claude/worktrees/v030-parity-gpt worker/opus/v030-native-init`
- `sed -n` / `tail` reads of `PROJECT_CONSTITUTION.md`, `AGENTS.md`,
  `.agent/skills/managing-sprints/SKILL.md`, and the v0.3/v0.4 sprint plan
- `find` / `ncdump -h` / Python netCDF4 inventory commands against
  `/mnt/data/canairy_meteo/runs/wps_cases` and AIFS roots
- `python -m py_compile proofs/v030/parity/metem_parity_scorer.py`
- d03 smoke self compare: PASS over 13 files
- full self compare command above: PASS over 507 files
- perturbed-copy command above: expected FAIL on `TT`

All substantive data scans and validation commands were run under
`taskset -c 0-3`.

## Files Changed

- `proofs/v030/parity/metem_parity_scorer.py`
- `proofs/v030/parity/metem_self_parity.json`
- `proofs/v030/parity/metem_perturbed_parity.json`
- `.agent/reviews/2026-06-02-gpt-v030-parity-harness.md`

## Unresolved Risks

- The local `worker/opus/v030-native-init` ref is behind the sprint-plan commit
  found in the main worktree. This branch contains the requested deliverables but
  not the plan file itself.
- AIFS expanded-field coverage is incomplete for the May21+ oracle cases.
- Current AIFS fields do not provide pressure-level winds, pressure-level
  humidity, most pressure-level temperature/height fields, soil fields, or
  skin/SST. The ingest implementation must add sources or make explicit,
  reviewed fallback decisions before it can pass this oracle honestly.
- The scorer validates the metgrid-equivalent artifact against WPS output; it
  does not implement or bless any ingest derivation.

## Next Decision Needed

Before S1/S3/S5 can close honestly, the manager needs a source-policy decision
for the flagged AIFS gaps, especially pressure-level U/V, pressure-level
humidity, missing pressure levels for TT/GHT, soil fields, `SKINTEMP`, `SOILHGT`,
and May expanded-field backfill.
