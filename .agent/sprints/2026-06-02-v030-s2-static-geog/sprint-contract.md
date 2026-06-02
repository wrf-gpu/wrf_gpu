# Sprint Contract — v0.3.0 S2 (static geog + projection)

Owner: **GPT**. CPU-only. File-disjoint from S1/S3/S4.

## Objective
Produce the static-geography + projection half of the metgrid-equivalent artifact:
terrain, land use, soil categories, green fraction/albedo/LAI, snow albedo,
orographic slope params, the land/sea mask, plus the projection-derived coordinate
and metric fields (lat/lon on mass/U/V/corner, map factors, Coriolis, rotation).
These are NOT interpolated from AIFS — they come from `geo_em.d0N.nc` (the geogrid
product) and the Lambert projection definition.

## Non-Goals
- NO atmospheric/soil-state decode (S1) or interp of AIFS fields (S3).
- NO re-running geogrid; consume the existing `geo_em.d0N.nc`.

## File Ownership (DISJOINT)
- `src/gpuwrf/init/static_geog.py` — reads `geo_em.d0N.nc`, extracts the 30 static
  geog/coord/mapfac field specs (schema `group` ∈ {coord, mapfac, geog2d, geog3d}).
- `src/gpuwrf/init/projection.py` — Lambert-conformal forward/inverse, map-factor
  (`msftx/y` family), Coriolis f = 2Ω sin(lat), rotation SINALPHA/COSALPHA; usable
  to DERIVE the coord/mapfac fields independently of geo_em for cross-check.
- Tests: `tests/init/test_static_geog.py`, `tests/init/test_projection.py`.

## Inputs
- Static geog: `/mnt/data/canairy_meteo/runs/wps_cases/<case>/l3/geo_em.d0N.nc`
  (51 vars; RECON.md §4). Same projection global attrs as met_em (Lambert,
  truelat1/2=25/30, stand_lon=−16.4, MMINLU MODIFIED_IGBP_MODIS_NOAH, NUM_LAND_CAT
  21, NUM_SOIL_CAT 16).
- Schema field specs with `source="geo_em"` (the S2 field set).
- met_em static block (RECON.md §2/§4) as the OUTPUT reference (the met_em files
  carry these copied-through; the artifact must match met_em, which == geo_em here).

## Acceptance Criteria
- Extracts every `source="geo_em"` schema field at the correct dims/stagger for
  d01/d02/d03 (coord, mapfac, geog2d, geog3d groups; 30 fields incl the X/Y map-factor
  variants MAPFAC_*X/*Y, the corner coords XLAT_C/XLONG_C, LANDMASK, SOILTEMP,
  OA1-4/OL1-4).
- `projection.py` reproduces XLAT_M/XLONG_M from the Lambert params to ≤1e-4 deg vs
  geo_em (independent derivation cross-check, not just a copy).
- Map factors MAPFAC_M reproduced to ≤1e-5 (rel) vs geo_em.
- Coriolis F reproduced to ≤1e-9 s⁻¹ vs geo_em.
- Category fields (LU_INDEX, LANDMASK, SCT_DOM, SCB_DOM) bit-exact vs geo_em
  (integers/flags — parity_tol=0).
- LANDUSEF(21) / SOILCTOP/CBOT(16) / monthly GREENFRAC/ALBEDO/LAI(12) extracted at
  the right category/month dims.

## Predeclared per-field parity tolerances (vs geo_em, == the met_em static block)
Use the schema `parity_tol`/`rel_tol`/`masked` per field (RECON.md §5):
XLAT*/XLONG*≤1e-4 deg; MAPFAC_*≤1e-5 rel; F≤1e-9, E≤1e-9; SINALPHA/COSALPHA≤1e-6;
HGT_M≤0.5 m; SOILTEMP≤0.5 K; LU_INDEX/LANDMASK/SCT_DOM/SCB_DOM exact;
LANDUSEF/SOILC*≤1e-4; GREENFRAC≤1e-3; ALBEDO12M≤0.5; LAI12M≤1e-2; SNOALB≤0.5;
VAR≤1.0/1e-3rel; CON/OA/OL≤1e-3.

## Validation Commands
```
JAX_PLATFORM_NAME=cpu taskset -c 0-3 python3 -m pytest tests/init/test_static_geog.py \
  tests/init/test_projection.py -q
```

## Proof Object
`proofs/v030/s2_static_geog_report.json`: per-field max-abs/rel error vs geo_em for
d01/d02/d03 across ≥3 cases, the independent-projection-derivation cross-check
numbers, and a PASS/FAIL per field at the predeclared tolerances.

## Risks
- Map-factor convention: WRF stores both isotropic (MAPFAC_M) and anisotropic
  (MAPFAC_MX/MY); derive both, match geo_em's storage exactly.
- Corner/stag lat-lon (XLAT_C/XLONG_C, XLAT_U/V) staggering offsets are easy to get
  half-a-cell wrong; validate against geo_em directly.
- Lambert with two true latitudes (25/30) — use the secant-cone formula, not tangent.

## Handoff Requirements
objective, files, commands, proof, the projection-parameter set actually used,
unresolved risks, the `projection.py` public API (S3 imports it read-only).
