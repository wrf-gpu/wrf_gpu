# CANARY all-7-islands nested geometry — Phase 1 coverage report

Status: **geogrid PASSED for all 9 domains; all 7 islands covered at 1 km with margin.**
Date: 2026-06-15. Phase 1 only (geometry + geogrid). metgrid/real/CPU/GPU = Phase 2 after manager review.

## Projection (unchanged from operational config — authority = operational l3 namelist.wps)
- map_proj = lambert
- ref_lat = 28.3, ref_lon = -16.4
- truelat1 = 25.0, truelat2 = 30.0, stand_lon = -16.4
- WPS earth radius 6370 km (validated: my LCC reproduced the operational d01/d02 geo_em corners to 1e-4 deg).

NOTE: the dispatch brief stated `ref_lat=28.535, ref_lon=-15.755`. That is WRONG. The authoritative
operational namelist.wps (and every operational geo_em CEN_LAT/CEN_LON / MOAD attr) is
`ref_lat=28.3, ref_lon=-16.4`. I used the namelist authority, as instructed.

## Geog data
- geog_data_path = <DATA_ROOT>/canairy_meteo/artifacts/wps_geog/WPS_GEOG_LOW_RES
- geog_data_res  = 'lowres' (d01), 'copdem30_canary+lowres' (d02..d09)  — same scheme as operational run.
- GEOGRID.TBL = GEOGRID.TBL.ARW (has the copdem30_canary / esa_worldcover tokens).
- Terrain sanity from geo_em (max HGT): Tenerife 3462 m, La Palma 2270 m, Gran Canaria 1840 m,
  La Gomera 1372 m, El Hierro 1299 m, Fuerteventura 562 m, Lanzarote 557 m — all physically correct.

## Design
- d01 9 km (94x60): UNCHANGED. Covers whole archipelago; contains d02 with >=6-cell margin all sides.
- d02 3 km (196x94): ENLARGED east+north from operational 160x67 so all 7 1-km nests sit >=4 parent
  cells inside, and so it still fits inside d01 with margin.
- 9 domains total: d01 -> d02 -> seven 1-km nests, ONE PER ISLAND.
  (Each island kept as its own nest rather than grouping, because spacing makes a grouped nest larger
  in total columns than two small nests; per-island nests stay compact.)
- parent_grid_ratio = 3, parent_time_step_ratio = 3 for every level. All (e_we-1) and (e_sn-1)
  divisible by 3 (clean nest alignment).
- feedback = 0 (one-way, matching operational).

## Per-domain table (values read back from produced geo_em files; cols = mass-grid (e_we-1)x(e_sn-1))

| dom | island        | dx (m) | e_we | e_sn | mass cols | parent | i_parent_start | j_parent_start | geo box (lat,lon SW -> NE) |
|-----|---------------|--------|------|------|-----------|--------|----------------|----------------|-----------------------------|
| d01 | (all)         | 9000   | 94   | 60   | 5,487     | d01    | 1              | 1              | (25.89,-20.72)->(30.65,-12.08) |
| d02 | (all)         | 3000   | 196  | 94   | 18,135    | d01    | 22             | 18             | (27.26,-18.76)->(29.79,-12.73) |
| d03 | Tenerife      | 1000   | 103  | 70   | 7,038     | d02    | 54             | 26             | (27.97,-17.12)->(28.58,-16.08) |
| d04 | Gran Canaria  | 1000   | 64   | 67   | 4,158     | d02    | 93             | 14             | (27.64,-15.92)->(28.23,-15.29) |
| d05 | La Palma      | 1000   | 58   | 55   | 3,078     | d02    | 20             | 42             | (28.39,-18.17)->(28.87,-17.59) |
| d06 | La Gomera     | 1000   | 40   | 40   | 1,521     | d02    | 43             | 24             | (27.91,-17.45)->(28.25,-17.06) |
| d07 | El Hierro     | 1000   | 40   | 40   | 1,521     | d02    | 18             | 10             | (27.52,-18.21)->(27.87,-17.82) |
| d08 | Fuerteventura | 1000   | 100  | 88   | 8,613     | d02    | 137            | 28             | (27.99,-14.58)->(28.78,-13.56) |
| d09 | Lanzarote     | 1000   | 76   | 76   | 5,625     | d02    | 154            | 58             | (28.80,-14.04)->(29.48,-13.26) |

- **TOTAL 1 km columns = 31,554** (target 20k-45k; was 144,801 in the wrong single-domain design — 4.6x smaller).
- **TOTAL all-domain columns = 55,176.**

## Per-island coverage table (island center -> nearest geo_em cell; margin = min distance to nest edge)

| island        | nest | center cell (i,j) | min edge margin (1 km cells) | land % in nest | fully encloses island bbox |
|---------------|------|-------------------|------------------------------|----------------|----------------------------|
| Tenerife      | d03  | (48,36)           | 32                           | 29.0           | yes |
| Gran Canaria  | d04  | (31,35)           | 30                           | 37.6           | yes |
| La Palma      | d05  | (30,32)           | 21                           | 22.8           | yes |
| La Gomera     | d06  | (21,21)           | 17                           | 24.1           | yes |
| El Hierro     | d07  | (21,24)           | 14                           | 17.4           | yes |
| Fuerteventura | d08  | (51,40)           | 40                           | 19.2           | yes (incl NE tip / Lobos) |
| Lanzarote     | d09  | (39,25)           | 25                           | 15.0           | yes (incl Chinijo / La Graciosa to ~29.48 N) |

All 7 islands: center >= 14 cells inside its 1 km nest; each nest contains real land (15-38%);
each nest geo box fully encloses the island's coastline bounding box. Every 1 km nest sits
>= 9 parent (3 km) cells inside d02. d02 sits >= 6 parent (9 km) cells inside d01.

## Honest coverage note
- All 7 islands ARE covered at 1 km with comfortable margin. The previous operational max_dom=5
  design actually missed FOUR islands at 1 km (not two): La Gomera (west of d03), El Hierro (south
  of d05), Fuerteventura, Lanzarote. This 9-domain design fixes all four.
- Land fractions (15-38%) confirm nests are island-centered, not open-ocean boxes.

## Parent-nest containment margins (parent 3 km cells), all >= 4 required
(computed from produced geo_em staggered dims and i/j_parent_start)
d03 W53 E107 S25 N44 | d04 W92 E81 S13 N57 | d05 W19 E156 S41 N33 | d06 W42 E139 S23 N56 |
d07 W17 E164 S9 N70 | d08 W136 E25 S27 N36 | d09 W153 E16 S57 N10.
Min = 9 (d07 south, El Hierro); next = 10 (d09 north, Lanzarote/Chinijo). All >= 4 — OK.

## Risks / decisions for the manager
1. **gwd_opt = 1 flagged, NOT changed.** Gravity-wave drag is normally for coarse grids; on 1-3 km
   nests it is usually off (gwd_opt=0). I left it at the operational value (1) per the "flag don't
   silently change" instruction. RECOMMEND setting gwd_opt=0 for d02..d09 (or globally) for Phase 2;
   awaiting manager call. (WRF applies gwd per-domain; current single scalar gwd_opt=1 applies to all.)
2. **d02 is now 196x94 = 18,135 cols** (was 160x67). Larger but necessary to host the two eastern
   nests with margin while staying inside d01. d01 (94x60) still contains it with >=6 cell margin.
   If a smaller d02 is preferred, the eastern islands could instead hang off a second 3 km nest, but
   that complicates the tree; single wide d02 is simplest and was verified.
3. **9 domains / max_dom=9.** More nests than the operational 5. Phase-2 real.exe + WRF must be built
   for max_dom>=9 (the install supports arbitrary max_dom via namelist; confirm at real.exe step).
4. **nproc_x/nproc_y removed** from namelist.input (operational had 7x4=28 for a 28-rank run). Phase 2
   12-rank CPU run will set its own decomposition; tiny nests (39x39) constrain max ranks per domain.
5. **time_step = 18 s** kept from operational (9 km parent, CFL-safe; 1 km nests get dt = 18/27 = 0.67 s
   via the 3x3x3 chain). Fine.
6. Terrain on d03 Tenerife maxes at 3462 m vs Teide's true 3715 m — expected smoothing from the
   lowres+copdem30 blend at 1 km; acceptable, matches operational behaviour.

## Files
- namelist.wps, namelist.input: this directory.
- geo_em.d01.nc .. geo_em.d09.nc: this directory (also in <DATA_ROOT>/wrf_downscale/canary_all7/wps_geo/).
- geogrid.log: this directory ("Successful completion of geogrid.exe").
- Working dir: <DATA_ROOT>/wrf_downscale/canary_all7/wps_geo/
