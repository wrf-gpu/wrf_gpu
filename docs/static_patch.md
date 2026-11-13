# Static-field patches: land use, soil, roughness, terrain

`gpuwrf patch` prepares land-surface and terrain experiments (for example "what if
this slope were grassland", "wetter soil on the island", "smoother terrain") on
private copies of a WRF case. It is a host tool: it never changes the forecast
program, and every patch is checked so that the edited case stays consistent
with what WRF's preprocessors would have produced. If a check fails, nothing is
written.

| Subcommand | What it changes | Output |
| --- | --- | --- |
| `patch land` | land-use class (with its dependent fields), soil class, soil moisture/temperature | new `wrfinput_dNN` + `wrflowinp_dNN` copies, `PATCH_PROTOCOL.json` |
| `patch z0` | Noah-MP roughness (`z0mvt`) on a cell mask | a surface-control file for `gpuwrf run --surface-controls` |
| `patch geo-em` | terrain height `HGT_M` (smooth, set, add, scale, replace from a DEM grid) | new `geo_em.dNN.nc` copies, `GEO_PATCH_PROTOCOL.json` |
| `patch check-real` | nothing (read only) | checks real.exe output made from a patched geo_em |
| `patch verify` | nothing (read only) | re-audits any edited case against its source files |

The simpler [initial-field overrides](initial_overrides.md) (`gpuwrf override`)
remain the tool for single-field edits such as SST or moisture.

## Land use and soil

```json
{"patches": [
  {"nest": 2, "kind": "landuse", "class": 16, "mask": [{"bbox": [28.2, 28.35, -16.75, -16.5]}, "land"], "dependents": "donor"},
  {"nest": 3, "kind": "landuse", "class": 16, "mask": [{"bbox": [28.2, 28.35, -16.75, -16.5]}, "land"], "dependents": "donor"},
  {"nest": 3, "kind": "soil", "field": "SMOIS", "op": "scale", "values": 1.2, "mask": [{"bbox": [28.2, 28.35, -16.75, -16.5]}, "land"]}
]}
```

```bash
JAX_PLATFORMS=cpu gpuwrf patch land --input-dir CASE --output-dir CASE_EDIT \
    --spec patch.json --namelist CASE/namelist.input
```

Masks are `all`, `land`, `water`, `class:N`, `{"bbox": [lat0, lat1, lon0, lon1]}`,
`{"polygon": [[lat, lon], ...]}`, a 0/1 array or `{"file": "cells.npy"}`; a list
of masks is their intersection. Use the same geographic mask on every nest that
covers the area, so parent and child stay consistent.

A land-use change sets `LU_INDEX`, `IVGTYP` and `LANDUSEF` (one-hot). The
vegetation climatology at the cell (`SHDMAX`, `SHDMIN`, `SHDAVG`, `VEGFRA`, `LAI`,
`ALBBCK`, `SNOALB`, and `VEGFRA`/`ALBBCK`/`LAI` in every `wrflowinp` record)
comes from WPS datasets, not from the class, so the spec must say what happens
to each of these fields:

- `"donor"`: copy them from the nearest unpatched cell of the target class in the
  same nest (all wrflowinp records too). The protocol records donor distances.
  If the nest has no cell of that class, the patch is refused.
- a value (scalar, per-soil-level list or full field) for that field,
- `"keep"`: leave the old climatology, only with a `dependent_keep` waiver.

`"dependents": "donor"` applies the donor rule to every required field.
`soil_class` (STAS 1–19, not 14 water or 16 land ice) optionally sets `ISLTYP`
and one-hot `SOILCTOP`/`SOILCBOT`; soil moisture is then checked against the new
class's porosity. Soil patches (`kind: soil`) edit `SMOIS` or `TSLB` on land
cells only, with `set`/`scale`/`add` and optional `layers` (0-based).

**Land/sea flips** (a land cell becomes class 17 water, or a water cell becomes
land) must patch the whole set that real.exe would have produced:

- land → water: `LANDMASK=0`, `XLAND=2`, `ISLTYP=14`, and the water values measured
  on real.exe output (`SMOIS=SH2O=1`, `ALBBCK=SNOALB=0.08`, vegetation 0,
  `SEAICE/CANWAT/SNOW=0`) are set automatically; `SST`, `TSK`, `TMN`, `TSLB` and
  the wrflowinp `SST` records need a donor (nearest water cell) or values;
- water → land: `soil_class` is required, and `SMOIS`, `TSLB`, `TSK`, `TMN` plus the
  vegetation fields need a donor (nearest land cell of the target class) or values.

Glacier (15) and lake (21) are refused as target and as source: their
initialisation belongs to real.exe and the lake model.

### Checks (fail closed)

| Check | What it requires | Waivable |
| --- | --- | --- |
| `lu_alias`, `landusef` | `IVGTYP == LU_INDEX`; patched `LANDUSEF` sums to 1 with argmax = class | no |
| `land_water_mask`, `soil_category` | `XLAND = 2 − LANDMASK`; water classes ⇔ `LANDMASK = 0`; water soil 14 ⇔ water | no |
| `flip_set` | every field of a land/sea flip patched as above | no |
| `soil_porosity`, `soil_liquid` | `SMOIS ≤ porosity(ISLTYP)`, `0 ≤ SH2O ≤ SMOIS` | the porosity cap only |
| `hard_ranges` | changed values finite; `SMOIS`, `SH2O` in [0, 1]; `TSLB`, `TSK`, `TMN`, `SST` in [180, 350] K | no |
| `field_ranges`, `vegetation_order` | other physical ranges (vegetation %, LAI, albedo); `SHDMIN ≤ VEGFRA ≤ SHDMAX` | yes |
| `nest_coverage` | a changed parent cell over a child's interior has patched child cells under it (the child never sees parent land fields) | yes |
| `nest_boundary_zone` | a changed child cell in the `spec_bdy_width` zone has the parent cell above it patched (that zone is forced from the parent) | yes |
| `base_state_untouched`, `untouched_elsewhere` | terrain, base state and atmosphere byte-identical; only declared fields on declared cells change | no |
| `nest_parent_missing`, `nest_set` | every nest's parent file is in the input directory; exactly the nests 1..max_dom of `--namelist` are present (`--namelist` is required for any non-empty spec) | no |

Soil patches see the land/sea mask after the land-use patches of the same spec,
so a cell flipped to water is no longer a soil cell. A waiver needs a reason and
is recorded as `WAIVED`. An empty spec `{}` copies every file byte for byte.

`gpuwrf patch verify --input-dir CASE --patched-dir CASE_EDIT --namelist CASE/namelist.input [--waiver CHECK=REASON]`
re-audits any edited case, trusting only the files: every differing cell counts
as changed, and the consistency, flip, range, porosity, untouched and nest checks
above run on it. It also fails when the source has a `wrflowinp_dNN` that the
edited case lacks (`aux_present`), and when a re-classed land cell keeps a
vegetation value that no unpatched cell of its new class has in the source
(`dependents_updated`, waivable with `dependent_keep`). It cannot know which
dependent policy was intended; that is recorded only in the patch protocol.

### Assembling a runnable case

`patch land` writes only the edited `wrfinput_dNN`, `wrflowinp_dNN` and the
protocol. Build the forecast directory from the unchanged source case plus these
files, for example:

```bash
mkdir CASE_RUN
cp CASE/namelist.input CASE/wrfbdy_d01 CASE_RUN/        # plus any .dat/table files the case carries
cp CASE_EDIT/wrfinput_d0* CASE_EDIT/wrflowinp_d0* CASE_RUN/
gpuwrf run --input-dir CASE_RUN --output-dir OUT --domains-from-namelist
```

`wrfbdy_d01` stays valid: land-surface patches do not change the lateral
boundaries. A terrain patch does change them, which is why terrain goes through
real.exe.

## Roughness (z0)

Under Noah-MP the roughness the surface layer sees comes from the `Z0MVT` table
by vegetation class, so a `ZNT` edit in wrfinput has no effect. `patch z0` writes
a traced [surface control](surface_params.md) instead: the same compiled
program runs any roughness values.

```bash
JAX_PLATFORMS=cpu gpuwrf surface-params basis --input-dir CASE_EDIT --output basis.npz
gpuwrf patch z0 --input-dir CASE_EDIT --basis basis.npz --spec z0.json --output z0.controls.npz
gpuwrf run --input-dir CASE_RUN --output-dir OUT --domains-from-namelist --surface-controls z0.controls.npz   # CASE_RUN assembled as above
```

with `z0.json` = `{"patches": [{"nest": 3, "mask": [{"bbox": [...]}, "land"], "factor": 2.0}]}`.
Factors multiply the table value on the mask (1 elsewhere). Where the table value is 0
(for example MODIS barren in the WN3 tables) a factor has no effect. The basis must come
from the same, possibly land-patched, case; a different `IVGTYP` is refused.

## Terrain

Terrain is never edited in wrfinput: the base state (`MUB`, `PB`, `PHB`, `ALB`,
`T_INIT`), the balanced initial state and the lateral boundaries all follow from
`HGT` inside real.exe. The route is:

1. `gpuwrf patch geo-em` writes new geo_em files (only `HGT_M` changes);
2. metgrid.exe (met_em files carry `HGT_M`) and real.exe run on them;
3. `gpuwrf patch check-real` checks the new wrfinput.

```json
{"patches": [
  {"nest": 2, "op": "smooth", "passes": 2, "taper_cells": 1, "mask": [{"bbox": [28.2, 28.35, -16.75, -16.5]}, "land"]},
  {"nest": 3, "op": "smooth", "passes": 4, "taper_cells": 3, "mask": [{"bbox": [28.2, 28.35, -16.75, -16.5]}, "land"]}
]}
```

```bash
gpuwrf patch geo-em --geo-dir GEO --output-dir GEO_EDIT --spec terrain.json --namelist namelist.input
gpuwrf patch check-real --geo-dir GEO_EDIT --real-dir REAL_EDIT --reference-dir REAL_ORIG --namelist namelist.input
```

`op` is `smooth` (1-2-1 passes, as WPS), `set`, `add` or `scale`; replace terrain
with a DEM already on the model grid via `set` + `file`. `taper_cells` blends the
result into the original over a ring around the mask. geo-em refuses changes on
water cells (`water_terrain`), on a child's spec + blend zone (`nest_blend_zone`:
wrf.exe replaces that zone with the parent terrain), on parent cells under a
child's spec + blend zone (`parent_under_child_zone`: wrf.exe blends them into the
child, and check-real would reject the run) and on parent cells over a child
interior without the child (`nest_coverage`); a missing parent file is never
accepted (`nest_parent_missing`). Edits on a domain with gravity-wave drag
(`gwd_opt` 1 or 3) are refused unless waived (`gwd_subgrid_orography`), because
geo_em `VAR`, `CON`, `OA1–4` and `OL1–4` keep the statistics of the old DEM; an
unknown `gwd_opt` counts as on. Pass `namelist.input` (it has `gwd_opt`,
`spec_bdy_width` and the nest geometry; `namelist.wps` lacks `gwd_opt`).

check-real verifies: `HGT == HGT_M`; the base state recomputed from `HGT` with
real.exe's formulas (`module_initialize_real.F`, hypsometric_opt 1/2) matches
the file (relative tolerance 2e-6 for MUB/PB/ALB/θ, 5e-5 for PHB; the unpatched
WN3 0227 files reproduce to ≤ 5.8e-7 and 2.0e-5, a 1 m terrain error moves MUB by
about 1e-4); `PHB(surface) = g·HGT`; positive mass and layer thickness; with a
reference run, land statics unchanged and the child spec + blend zone terrain and
the parent cells under it unchanged. On a multi-nest case without
`--reference-dir`, that zone check fails, because it cannot be proven.

## Compile cost and what changes the program

| Patch | Effect on the compiled forecast program |
| --- | --- |
| soil moisture / temperature, `VEGFRA`, `TMN`, `TSK`, `SST`, wrflowinp records, `z0` control | none: traced inputs, the cached program is reused |
| land-use class, soil class, land/sea flip, `SHDMAX` (any land-use change with donor climatology) | new program: these Noah-MP static fields are compile constants; expect one cold compile and one new compile cache per variant |
| terrain (`HGT` via real.exe) | new program: terrain and grid metrics are compile constants |

This follows from the code (the Noah-MP static bundle and the grid terrain are
hashed into the compile key). It is confirmed for a land-use class patch only
(below); soil class, land/sea and terrain patches have not run on the GPU. Plan the number of land-use or terrain variants with the cold
compile time of the case in mind. The tool itself costs about 1–4 s and under
1 GB of host memory for the WN3 three-nest case; it adds nothing to the model's
time step.

## Limits

- Checks are about consistency, not about whether an experiment is physically
  sensible. A donor copies climatology from a nearby cell of the class; it does not
  derive it.
- Inputs must use MODIS 21-class land use (`MODIFIED_IGBP_MODIS_NOAH`).
- The terrain checks cover real.exe's base state; the full initial-state balance
  and `wrfbdy` remain real.exe's responsibility.
- A terrain patch changes only `HGT_M`. The subgrid orography statistics (`VAR`,
  `CON`, `OA1–4`, `OL1–4`) and `SOILTEMP` in geo_em stay at the old DEM; regenerate
  them with geogrid if the experiment depends on gravity-wave drag.
- One land-use patch (barren land on part of the 1 km Tenerife nest) has run on the GPU for 3 hours with finite
  output, from an empty compile cache. Its compile keys (checked on CPU): domain 1 and domain 2 unchanged, the
  1 km nest and the fused nest program changed. On the GPU both the domain-1 program and the fused nest program
  were compiled because the cache was empty; the domain-1 program is identical to the unpatched one (same program
  hash), the fused nest program is new. Reusing the unpatched domain-1 binary from a filled cache was not exercised.
  No patched forecast has been compared with CPU-WRF. Soil-class, land/sea and terrain patches have not run on the
  GPU, nor has a `z0` control from `gpuwrf patch z0`.

Return to [Tune WRF for your region](regional_optimization.md).
