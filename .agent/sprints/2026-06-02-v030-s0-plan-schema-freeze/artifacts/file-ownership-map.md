# v0.3.0 file-ownership map (FROZEN by S0)

Disjoint ownership so S1, S2, S3, S4 run in parallel without colliding. The schema
module is the ONLY shared interface; it is frozen by S0 and edited by no lane.

| Path | Owner | Notes |
|---|---|---|
| `src/gpuwrf/init/metgrid_schema.py` | **S0 (frozen)** | read-only for all lanes; additive optional specs only with manager sign-off |
| `src/gpuwrf/init/__init__.py` | **S0 (frozen)** | re-exports schema; lanes append their own exports via separate review |
| `src/gpuwrf/init/forcing_decode.py` | **S1** | GRIB2 → per-field source arrays on the AIFS 0.25° grid |
| `src/gpuwrf/init/aifs_grib.py` | **S1** | eccodes/cfgrib reader + Vtable.AIFS_PURE mapping + surface-level assembly + dewpoint→q |
| `src/gpuwrf/init/static_geog.py` | **S2** | geo_em.d0N reader → coord/mapfac/landuse/soil/orog static fields |
| `src/gpuwrf/init/projection.py` | **S2** | Lambert forward/inverse + map-factor + Coriolis derivation (shared-read by S3) |
| `src/gpuwrf/init/interp_metgrid.py` | **S3** | horizontal interp kernels (four_pt/sixteen_pt/avg/nn/search) + masking + PRES build |
| `src/gpuwrf/init/metgrid_assemble.py` | **S3** | drives S1+S2 outputs through interp → `MetEmArtifact` |
| `src/gpuwrf/init/metem_writer.py` | **S3** | `MetEmArtifact` → met_em-format NetCDF (Time axis, global attrs, FLAG_*) |
| `proofs/v030/parity/**` | **S4 (GPT lane)** | comparator harness + per-variable parity report (DO NOT TOUCH from other lanes) |
| `src/gpuwrf/init/namelist_v030_check.py` | **S5** | P2-2 reject unsupported namelist options for the ingest path |
| `proofs/v030/RECON.md`, `recon_inventory.*` | **S0** | recon artifacts |

## Shared-read (no lane writes these)
- `src/gpuwrf/init/metgrid_schema.py` — the frozen contract.
- `src/gpuwrf/contracts/grid.py` (Projection/GridSpec) — alignment reference only.

## Merge order
S1, S2, S3 land file-disjoint; S4 (parity) lands independently in `proofs/v030/parity/`;
S5 integrates + runs the ≥10-case gate. Manager merges; no two lanes edit a shared file.

## Projection.py shared-read note
S3's interp needs target lat/lon (from S2's `projection.py`) and source lat/lon (AIFS
0.25° grid, S1). To keep ownership disjoint: S2 OWNS `projection.py`; S3 only imports
it. If S3 needs a projection change, request it from S2 (do not edit).
