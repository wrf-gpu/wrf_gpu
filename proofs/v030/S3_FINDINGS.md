# v0.3.0 S3 — metgrid interp kernels + assembly + writer (findings)

Date 2026-06-02. Branch `worker/opus/v030-s3` (base `worker/opus/v030-s0`, 072fbbc).
Owner: Opus MAX. CPU-only, file-disjoint from S1/S2/S4.

## What was built
- `src/gpuwrf/init/interp_metgrid.py` — vectorized JAX port of WPS
  `metgrid/src/interp_module.F`: `oned`, `nearest_neighbor`, `four_pt`
  (bilinear), `sixteen_pt` (overlapping-parabolic), `four_pt_average`,
  `sixteen_pt_average`, `wt_four_pt_average`, `wt_sixteen_pt_average`,
  `search_extrap`, the `+`-chain `interp_sequence` dispatcher (first-applicable
  fall-through), the source-cell mask predicate (`<`/`>`/` ` relational), and the
  `regular_ll` source-grid `lltoxy` (`latlon_to_source_xy`) + per-slab driver
  `interp_field_to_grid`.
- `src/gpuwrf/init/metgrid_assemble.py` — `assemble_met_em(...)` (the S5 entry
  point). Drives S1 forcing + S2 static through the kernels onto the target
  C-grid per stagger (M/U/V), builds PRES (index0=PSFC, 1..13=isobaric consts),
  the surface metgrid level (TT[0]=2t, UU[0]=10u, GHT[0]=SOILHGT, ...), the
  soil packing (ST/SM 3D stack bottom-band-first; SOIL_LAYERS=[40,10]), applies
  soil water-masking + `fill_missing`, and copies S2 geog through.
- `src/gpuwrf/init/metem_writer.py` — `write_met_em(...)` → met_em-format NetCDF
  (FieldType=104, MemoryOrder XYZ/`XY `, stagger, sr_x/y, Times char array, full
  global + FLAG_* block).
- Tests `tests/init/test_interp_metgrid.py` (24) + `tests/init/test_metem_writer.py` (6).
- Oracle `proofs/v030/s3_oracle/` — REAL `interp_module.F` compiled into
  `liboracle.so` (stub `module_debug` to avoid the MPI/parallel_module chain;
  `-D_METGRID` for the queue `q_data` variant) + a ctypes bridge. The JAX port is
  graded against the genuine Fortran routines, not a Python re-derivation.

## Oracle parity (proofs/v030/s3_interp_report.json)
- Every kernel + `oned`: max rel err 5e-8 … 3.2e-7 vs interp_module.F → **all < 1e-6** (AC).
  Residual is the Fortran single-precision (`real`) rounding vs JAX fp64.
- `sixteen_pt+four_pt+average_4pt` chain (TT/UU/VV/GHT/SPECHUMD): 2.0e-7 rel.
- Masked-soil chain `sixteen_pt+four_pt+wt_average_4pt+wt_average_16pt+search`
  with `interp_mask=LANDSEA(0)` equality exclusion: 1.4e-7 rel, finite/missing
  classification 100% agreement; pure-search (land island in water) tie-break
  matches; masked `four_pt` decline (straddling a masked column) matches.
- Assemble validates for d01/d02/d03; PRES build, GHT surface, soil packing OK;
  UU on west_east_stag, VV on south_north_stag (C-grid) — shapes correct.
- Writer structural diff = **0** vs real `met_em.d01.2026-04-28_18:00:00.nc`
  (dims + per-var FieldType/MemoryOrder/stagger + projection/FLAG_* globals).

## Field accounting (70 oracle vars)
- 19 forcing-derived (TT/UU/VV/GHT/SPECHUMD/PRES/PSFC/PMSL/SOILHGT/SKINTEMP/
  LANDSEA/DEWPT/ST/SM/SOIL_LAYERS/ST000010/ST010040/SM000010/SM010040) — S3.
- 46 geo_em static/coord/mapfac — copied through from S2.
- 5 omitted by schema design (Times + SINALPHA_U/V + COSALPHA_U/V).

## Faithfulness notes / honest caveats
- `sixteen_pt`'s `n/=16` averaging branch is DEAD in interp_module.F (`n` is
  always 16); intentionally omitted — verified it never fires.
- `search_extrap` is realized as "nearest usable cell within the L1-BFS depth,
  min Euclidean distance" — mathematically identical to the Fortran BFS for the
  result (validated). For fine Canary targets (1/3/9 km ≪ 27 km source) search +
  the `_average` coarsening branches are essentially inactive; they fire only for
  masked-water soil near coasts, which the tests do exercise.
- The masked-`four_pt`/`sixteen_pt` "decline if any stencil cell unusable" vs the
  `*_average` "partial-weight" behavior is reproduced exactly per Fortran.
- The `wt_four_pt_average` y-low boundary test has a known Fortran typo
  (`ifx < start_y`); reproduced behaviorally. Inactive for fine targets.

## What S5 must wire
- Call `assemble_met_em(domain, valid_time, projection, forcing, static,
  target_grid, source_grid)`:
  - `forcing` = `ForcingFields` from S1 (AIFS arrays on the SOURCE 0.25° grid:
    3D `(13, ny_src, nx_src)` in 1000..50 hPa order; 2D `(ny_src, nx_src)`).
  - `static` = dict of S2 geo_em arrays ON THE TARGET GRID, keyed by met_em name.
  - `target_grid` = `TargetGrid` of S2 lat/lon per stagger (M/U/V).
  - `source_grid` = `LatLonSourceGrid` (AIFS: lon0=0, dlon=0.25, lat0=90,
    dlat=-0.25, nx=1440, ny=721, global_wrap=True).
  - `projection` = `MetgridProjection` (per-domain dims, RECON §2).
- The end-to-end met_em VALUE parity vs the oracle is S4's gate (S3 grades the
  operators only). If S5 wants the SINALPHA_U/V + COSALPHA_U/V vars, add them as
  optional schema specs (manager sign-off per the freeze policy).
- Build the oracle once: `bash proofs/v030/s3_oracle/build.sh` (gfortran at
  `/home/enric/miniconda3/envs/wrfbuild/bin`); the test fixture auto-builds.

## Validation commands
```
JAX_PLATFORM_NAME=cpu JAX_COMPILATION_CACHE_DIR="" PYTHONPATH=src taskset -c 0-3 \
  python3 -m pytest tests/init/test_interp_metgrid.py tests/init/test_metem_writer.py -q
# proof regen:
JAX_PLATFORM_NAME=cpu PYTHONPATH=src taskset -c 0-3 python3 proofs/v030/s3_oracle/gen_proof.py
```
30 passed. (The benign XLA AOT-cache machine-feature warnings come from the repo
package cache; harmless, silenced with JAX_COMPILATION_CACHE_DIR="".)
