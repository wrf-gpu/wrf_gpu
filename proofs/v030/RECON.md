# v0.3.0 S0 — Input recon + TBD resolution

Date 2026-06-02. Branch `worker/opus/v030-s0` (from `worker/opus/v030-native-init`).
CPU-only, read-only recon. Machine-readable companion: `proofs/v030/recon_inventory.json`
(produced by `proofs/v030/recon_inventory.py`).

This recon resolves the two open input TBDs and traces the **complete forcing
chain** that produced the v0.3.0 oracle. It is the binding factual basis for the
frozen schema (`src/gpuwrf/init/metgrid_schema.py`) and the S1–S5 contracts
(`.agent/sprints/2026-06-02-v030-*`).

---

## 0. The forcing chain (now fully traced — the central recon result)

```
 AIFS global GRIB2 (0.25°, 1440×721, 13 isobaric + sfc + 2 soil layers)
   step_NNN.grib2  ──ungrib(Vtable.AIFS_PURE)──▶  AIFS:YYYY-MM-DD_HH (WPS intermediate)
                                                          │
                                  geo_em.d0{1..5}.nc ◀── geogrid (static geog)
                                                          │
                                  ──metgrid(METGRID.TBL.ARW)──▶  met_em.d0{1..5}.*.nc  (THE ORACLE)
                                                                        │
                                                          ──real.exe──▶ wrfinput/wrfbdy  (v0.4.0 consumes)
```

v0.3.0's native ingest must reproduce the boxed step: **(raw AIFS GRIB + static
geog) → met_em-equivalent**, gated against the 13-case `met_em.*` oracle. v0.4.0
then reproduces `real.exe` from the met_em-equivalent artifact.

**Pivotal correction vs the sprint plan's "Raw forcing: AIFS at
`/mnt/data/canairy_meteo/data/aifs_single*`":** those `aifs_single*.nc` files are
a *surface-only derived product* (10m/100m wind, 2m T, MSLP, sfc P, precip,
cloud, LW/SW) and CANNOT supply the 13-level 3D atmosphere met_em needs. The
**true raw forcing for S1 is the per-case 3D GRIB2** at
`/mnt/data/canairy_meteo/runs/wps_cases/<case>/ungrib/step_NNN.grib2`
(323 MB each, 13 isobaric levels of t/u/v/q/gh + surface + 2 soil layers). S1 must
read GRIB2 (via `cfgrib`/`eccodes`, both present), not the surface NetCDF.

---

## 1. TBD resolution

### TBD-a — geo_em static geog: **RESOLVED, PRESENT**
`geo_em.d0{1..5}.nc` are co-located with the oracle in every case:
`/mnt/data/canairy_meteo/runs/wps_cases/<case>/l3/geo_em.d0N.nc`.
51 variables, including the full static-geog set S2 needs (see §4). The met_em
files ALSO carry these static fields (geogrid copies them through), so either is
a valid reference; **S2 should match geo_em as primary** (it is the geogrid
product) and cross-check against the met_em static block.

### TBD-b — WPS / metgrid source on disk: **RESOLVED, PRESENT (source + built binaries)**
Full WPS source tree AND compiled binaries are on disk:
`~/src/canairy_meteo/Gen2/artifacts/wrf_src/WPS/`
- `install_gen2_dmpar/bin/metgrid.exe` (+ geogrid.exe/ungrib.exe in tree)
- `metgrid/METGRID.TBL` → `METGRID.TBL.ARW` (the active per-field interp spec)
- `metgrid/src/{interp_module.F, interp_option_module.F, process_domain_module.F,
  ...}` — the EXACT interp algorithm for S3 to port, not "work from docs".
- The GRIB→met_em map: `~/src/canairy_meteo/Gen2/configs/Vtable.AIFS_PURE`.
- WPS run drivers: `~/src/canairy_meteo/Gen2/scripts/run_wps_case.py`,
  `run_ungrib_case.py`, `run_wps_all_cases.sh`.

Consequence: S3 ports a **known algorithm** (metgrid `interp_module.F`), and S4
can, if needed, RE-RUN `metgrid.exe` to regenerate oracle / probe edge cases.

---

## 2. The oracle: `met_em.*` structure (binding)

13 `wps_cases/<case>/l3/` dirs, each with **13 timestamps × 5 domains** (d01–d05),
6-hourly (`interval_seconds=21600`), 72 h span. **70 variables** per file.
Operational target = **d01/d02/d03** (d04/d05 exist but are not the v0.3.0 gate).
WPS/metgrid version: **V4.6.0** (`TITLE` global attr). io_form_metgrid=2 (NetCDF).

### Grid / projection (global attrs — IDENTICAL projection family across domains)
| attr | d01 | d02 | d03 |
|---|---|---|---|
| MAP_PROJ | 1 (Lambert conformal) | 1 | 1 |
| TRUELAT1 / TRUELAT2 | 25.0 / 30.0 | 25.0 / 30.0 | 25.0 / 30.0 |
| STAND_LON | −16.4 | −16.4 | −16.4 |
| MOAD_CEN_LAT | 28.30001 | 28.30001 | 28.30001 |
| DX / DY (m) | 9000 / 9000 | 3000 | 1000 |
| west_east × south_north (mass) | 93×59 | 159×66 | 93×75 |
| GRID_DIMENSION (stag) | 94×60 | 160×67 | 94×76 |
| grid_id / parent_id / parent_grid_ratio | 1/1/1 | 2/1/3 | 3/2/3 |
| i_parent_start / j_parent_start | 1/1 | 24/20 | 52/20 |

Other binding global attrs (constant across domains): `GRIDTYPE="C"`, `DYN_OPT=2`,
`POLE_LAT=90`, `POLE_LON=0`, `MMINLU="MODIFIED_IGBP_MODIS_NOAH"`, `NUM_LAND_CAT=21`,
`ISWATER=17`, `ISLAKE=21`, `ISICE=15`, `ISURBAN=13`, `ISOILWATER=14`,
`NUM_METGRID_SOIL_LEVELS=2`. FLAG_* attributes (real.exe reads these) all = 1:
FLAG_METGRID, FLAG_SOIL_LAYERS, FLAG_PSFC, FLAG_SM000010/SM010040/ST000010/ST010040,
FLAG_SLP, FLAG_SH, FLAG_SOILHGT, FLAG_MF_XY, FLAG_LAI12M.

### Dimensions
`Time`(UNLIMITED=1), `DateStrLen`=19, `west_east`/`south_north` (mass),
`west_east_stag`/`south_north_stag` (= mass+1), `num_metgrid_levels`=14,
`num_st_layers`=2, `num_sm_layers`=2, plus geog category dims `z-dimension0012`
(=12, monthly), `z-dimension0016` (=16, soil cat), `z-dimension0021` (=21, land cat).

### Vertical levels (the 14 `num_metgrid_levels`)
**Surface + 13 isobaric**, top→down order as stored:
`[~100080 (sfc), 100000, 92500, 85000, 70000, 60000, 50000, 40000, 30000, 25000,
20000, 15000, 10000, 5000]` Pa. (Index 0 is the model-surface level; indices 1–13
are the standard isobaric levels 1000…50 hPa.) GHT/PRES/TT/UU/VV/SPECHUMD live on
all 14.

### Variable groups (70 total — full per-variable detail in recon_inventory.json)
- **3D atmos (num_metgrid_levels, stagger):** `PRES`(M), `GHT`(M), `TT`(M),
  `SPECHUMD`(M), `UU`(U-stag: west_east_stag), `VV`(V-stag: south_north_stag).
- **Soil (num_st/sm_layers):** `ST`, `SM`, `SOIL_LAYERS` (layer thickness, =[40,10]).
  Plus the named-layer 2D fields `ST000010, ST010040, SM000010, SM010040`.
- **Surface 2D:** `PSFC, PMSL, SOILHGT, SKINTEMP, LANDSEA, DEWPT`.
- **Static geog (copied from geo_em):** `HGT_M`-equivalent via `SOILHGT`,
  `LANDUSEF`(21), `LU_INDEX`, `SOILCTOP/SOILCBOT`(16), `SCT_DOM/SCB_DOM`,
  `GREENFRAC/ALBEDO12M/LAI12M`(12-monthly), `SNOALB`, `CON/VAR/OA1-4/OL1-4`
  (orographic slope), `SNOALB`, and the coordinate/metric fields
  `XLAT_M/XLONG_M, XLAT_U/XLONG_U, XLAT_V/XLONG_V, CLAT/CLONG, MAPFAC_*`.
- **NOTE on `DEWPT`:** metgrid emits a `WARNING: Entry in METGRID.TBL not found
  for field DEWPT` (it is not in METGRID.TBL.ARW), so DEWPT uses the metgrid
  *default* interp (nearest_neighbor h / linear_log_p v). It is a 2D surface field
  here (the Vtable maps 2d→DEWPT at level 103). real.exe uses SPECHUMD, not DEWPT,
  for moisture, so DEWPT is low-priority for parity (record it; loose tol).

---

## 3. Raw AIFS GRIB2 (the S1 input) — variable map

Source: `/mnt/data/canairy_meteo/runs/wps_cases/<case>/ungrib/step_NNN.grib2`.
Grid: `regular_ll`, **0.25° global, Ni=1440 Nj=721**, first point (90N,180E),
last (90S,179.75E) — i.e. 0..359.75°E increasing, 90N→90S. 78 GRIB messages/step.
6-hourly steps 000…072.

### What AIFS provides (GRIB shortName → typeOfLevel → met_em via Vtable.AIFS_PURE)
| AIFS GRIB shortName | level type | levels | → met_em | met_em units |
|---|---|---|---|---|
| `t` | isobaricInhPa | 13 (50–1000 hPa) | TT (3D) | K |
| `u` | isobaricInhPa | 13 | UU (3D, U-stag) | m s⁻¹ |
| `v` | isobaricInhPa | 13 | VV (3D, V-stag) | m s⁻¹ |
| `q` | isobaricInhPa | 13 | SPECHUMD (3D) | kg kg⁻¹ |
| `gh` | isobaricInhPa | 13 | GHT (3D) | m |
| `2t` | heightAboveGround 2 m | 1 | TT@2m (sfc) | K |
| `2d` | heightAboveGround 2 m | 1 | DEWPT@2m | K |
| `10u` | heightAboveGround 10 m | 1 | UU@10m | m s⁻¹ |
| `10v` | heightAboveGround 10 m | 1 | VV@10m | m s⁻¹ |
| `sp` | surface | 1 | PSFC | Pa |
| `msl` | meanSea | 1 | PMSL | Pa |
| `lsm` | surface | 1 | LANDSEA | 0/1 |
| `skt` | surface | 1 | SKINTEMP | K |
| `orog` | surface | 1 | SOILHGT | m |
| `st` (param 2) | depthBelowLandLayer 0,10 | 2 | ST000010, ST010040 | K |
| `unknown` (D2/C0/P192) | depthBelowLandLayer 0,10 | 2 | SM000010, SM010040 | fraction |

The Vtable maps the two soil layers by depth band: 0–1000 mm→`*000010`,
1000–4000 mm→`*010040` (cm in the met_em names). The `unknown` soil-moisture
shortName is the ECMWF volumetric soil water (GRIB2 discipline 2 / category 0 /
parameter 192); decode by the explicit Vtable triplet, not by shortName.

### GAPS / derivations (flagged for S1/S4 — these are met_em fields NOT directly in AIFS)
1. **PRES (3D pressure on the 14 metgrid levels)** is `derived=yes` in METGRID.TBL:
   built from the isobaric level values + `PSFC` for the surface level +
   `vertical_index` against TT. S1/S3 must construct PRES, not read it.
2. **The surface (k=0) metgrid level** = the 2 m / 10 m / surface fields packed as
   the lowest level: TT[0]←2t, UU[0]←10u, VV[0]←10v, SPECHUMD[0]←derived from 2d
   (dewpoint→q), GHT[0]←SOILHGT (orog), PRES[0]←PSFC. (This is metgrid's
   surface-level assembly; confirm exact recipe against `process_domain_module.F`.)
2b. **GHT units = OK as-is.** AIFS `gh` is reported in `gpm` (geopotential meters),
   which equals met_em GHT directly (no ÷g). Verified: AIFS gh@50hPa max ≈ 20997 gpm
   vs met_em GHT@50hPa ≈ 20786 m (consistent post-interp). S1 must NOT divide by g.
3. **SPECHUMD at the surface level** is derived from the 2 m dewpoint `2d` (no
   surface q in AIFS). RH/dewpoint→specific-humidity conversion (Bolton/Tetens as
   WPS uses) — flag the exact formula as an S1 oracle item.
4. **TAVGSFC, SST, SEAICE, SNOW** — not in this AIFS Vtable; metgrid leaves them
   unset (their FLAG_* are absent/0). Not required for the Canary namelist; the
   P2-2 namelist checker should confirm the run does not request them.
5. **Static geog fields in met_em** (LANDUSEF, SOILC*, GREENFRAC, etc.) do NOT come
   from AIFS — they are copied from `geo_em` by geogrid/metgrid. That is the S2
   lane, not S1.

---

## 4. Static geog `geo_em.d0N.nc` (the S2 reference) — 51 variables
Dims add `land_cat`=21, `soil_cat`=16, `month`=12 (vs met_em's z-dimension* names).
Field set S2 must reproduce (all on the metgrid grid, projected):
- **Coordinates/metrics:** `XLAT_M/XLONG_M, XLAT_U/XLONG_U, XLAT_V/XLONG_V,
  XLAT_C/XLONG_C, CLAT/CLONG`; `MAPFAC_M/U/V` and the X/Y variants
  `MAPFAC_MX/MY, MAPFAC_UX/UY, MAPFAC_VX/VY`; Coriolis `E, F`; rotation
  `SINALPHA, COSALPHA` (+ U/V-stag variants).
- **Land/soil:** `LANDMASK, LU_INDEX, LANDUSEF`(21), `SOILCTOP/SOILCBOT`(16),
  `SCT_DOM/SCB_DOM, SOILTEMP`.
- **Surface props:** `HGT_M`(terrain), `ALBEDO12M/GREENFRAC/LAI12M`(12-monthly),
  `SNOALB`.
- **Orographic slope (gravity-wave drag):** `CON, VAR, OA1–OA4, OL1–OL4`.
Same projection global attrs as met_em (Lambert, truelat 25/30, stand_lon −16.4,
MMINLU MODIFIED_IGBP_MODIS_NOAH, NUM_LAND_CAT 21).

---

## 5. metgrid interp spec (binding for S3 + S4 tolerances)
From `METGRID.TBL.ARW`. The metgrid **defaults** (interp_option_module.F:121-122)
for any field WITHOUT an explicit entry: horizontal `nearest_neighbor`, vertical
`linear_log_p`. Explicit per-field entries (the ones that matter):

| met_em field | interp_option (horizontal) | masking | fill_missing | mandatory |
|---|---|---|---|---|
| TT, UU, VV, GHT, SPECHUMD, PMSL, RH | `sixteen_pt+four_pt+average_4pt` | none | 0. | TT/UU/VV yes |
| PRES | derived (PRESSURE + PSFC@sfc + vertical_index vs TT) | — | — | yes |
| PSFC, SOILHGT | `four_pt+average_4pt` | none | — | — |
| SKINTEMP | `sixteen_pt+four_pt+wt_average_4pt+wt_average_16pt+search` | both | 0. | — |
| ST*/SM* (soil) | `sixteen_pt+four_pt+wt_average_4pt+wt_average_16pt+search` | masked=water, interp_mask=LANDSEA(0) | 1. | — |
| LANDSEA | `nearest_neighbor` | none | −1. | — |
| SOILCAT/VEGCAT | `nearest_neighbor` | none | 0. | — |
| DEWPT | default (nearest_neighbor h / linear_log_p v) | none | (default) | — |

`UU`/`VV` carry `is_u_field/is_v_field=yes` and `output_stagger=U/V` — they are
de-staggered/re-staggered to the C-grid U/V points. Vertical: isobaric→isobaric is
identity (same 13 levels); only the **surface level assembly** + **PRES
construction** involve vertical logic. The horizontal interp is source-0.25°-grid →
target Lambert grid; for these Canary domains the target is finer than 0.25° (9/3/1
km ≪ 27 km), so it is an UPSAMPLING/interp (each target cell maps inside one source
cell quad — the `sixteen_pt`/`four_pt` bicubic/bilinear stencil dominates; `search`
and the `average_4pt` coarsening branches are inactive for fine targets).

---

## 6. Recon-driven decisions baked into the schema + contracts
- Schema vertical = **14 metgrid levels (1 surface + 13 isobaric)**, fixed order,
  stored top-of-list = surface (matches met_em index 0 = surface).
- Schema grid = Lambert, C-grid, with mass/U/V/corner lat-lon + map factors +
  Coriolis from geo_em; per-domain dims as in §2.
- Schema is **NetCDF-compatible field-for-field with met_em** (same names, dims,
  staggering, units) so S4 parity is a direct variable-by-variable compare and so
  the v0.4.0 native-real consumer reads a met_em-faithful structure.
- S1 reads **GRIB2** (the 3D source), NOT `aifs_single*.nc`.
- S3 ports the **metgrid `interp_module.F`** algorithm (available on disk), not a
  generic interp.
- S4 may **re-run `metgrid.exe`** for oracle regeneration / edge probing.
