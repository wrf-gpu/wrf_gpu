# Sprint Contract — v0.3.0 S3 (interpolation kernels + artifact assembly)

Owner: **Opus MAX**. GPU-eligible (interp is the heavy kernel — JAX, may run on GPU;
ONE GPU job at a time). File-disjoint from S1/S2/S4.

## Objective
Port the WPS metgrid horizontal-interpolation kernels (source 0.25° AIFS grid →
target Lambert C-grid), staggering-aware, with the per-field masking and
missing-value/fill policy from `METGRID.TBL.ARW`; build the derived PRES field;
assemble S1 (forcing) + S2 (static geog) outputs into a `MetEmArtifact` and serialize
it to met_em-format NetCDF. This is the core of the metgrid equivalent.

## Non-Goals
- NO vertical re-interpolation between pressure levels (isobaric→isobaric is identity;
  only the surface-level assembly + PRES build involve the vertical, and the surface
  assembly is S1's). NO time interpolation in v0.3.0 (forcing is at the metgrid
  output cadence; 6-hourly steps map 1:1 to met_em timestamps).
- NO GRIB decode (S1), NO geo_em read (S2), NO parity comparator (S4).

## File Ownership (DISJOINT)
- `src/gpuwrf/init/interp_metgrid.py` — the kernels: `nearest_neighbor`, `four_pt`
  (bilinear), `sixteen_pt` (bicubic, via the 1-D Lagrange `oned`), the `_average`
  coarsening branches, the `wt_average` weighted variants, and the `search` extrap;
  the `+`-chained `interp_sequence` dispatcher; mask application (land/water/both via
  LANDSEA) and `fill_missing`.
- `src/gpuwrf/init/metgrid_assemble.py` — maps each schema field to its
  `interp_option`, drives S1+S2 arrays through the kernels onto the target grid,
  builds PRES (PRESSURE + PSFC@sfc + vertical_index vs TT), re-staggers UU→U pts /
  VV→V pts, returns a validated `MetEmArtifact`.
- `src/gpuwrf/init/metem_writer.py` — `MetEmArtifact` → NetCDF with the met_em Time
  axis, dimension names, per-variable attrs (FieldType/MemoryOrder/stagger/units/
  description), and all global attrs + FLAG_* (RECON.md §2).
- Tests: `tests/init/test_interp_metgrid.py`, `tests/init/test_metem_writer.py`.

## Inputs
- ALGORITHM REFERENCE (port this, do not invent): WPS
  `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_src/WPS/metgrid/src/interp_module.F`
  (functions: `interp_sequence`, `four_pt`, `sixteen_pt`, `four_pt_average`,
  `sixteen_pt_average`, `wt_four_pt_average`, `wt_sixteen_pt_average`,
  `nearest_neighbor`, `search_extrap`, `oned`) and
  `metgrid/src/process_domain_module.F` (the field-loop + masking + PRES build).
- Interp spec: `METGRID.TBL.ARW` (RECON.md §5).
- Schema: `gpuwrf.init.metgrid_schema` (field→interp_option, dims, stagger, masking).
- Source grid: AIFS 0.25° lat-lon (from S1). Target grid lat/lon + map factors:
  S2's `projection.py` (read-only import).

## Acceptance Criteria
- Each kernel reproduces its `interp_module.F` counterpart to ≤1e-6 (rel) on a
  unit-test grid with a known analytic field (independent oracle, fp64).
- The `+`-chain dispatcher selects the first applicable method exactly as
  `interp_sequence` does (e.g. `sixteen_pt+four_pt+average_4pt` falls back four→avg
  at boundaries).
- Masking: soil ST/SM masked to water=LANDSEA(0) with fill_missing per TBL; SKINTEMP
  masked=both; LANDSEA/SOILCAT nearest_neighbor.
- PRES built correctly: surface level = PSFC, isobaric levels = the level constants,
  matching met_em PRES to its parity_tol.
- UU on `west_east_stag`, VV on `south_north_stag` (C-grid), de/re-staggering correct.
- `metem_writer` output re-opens with the same dims/attrs/FLAG_* as a real met_em
  file (structural diff = 0; values are the S4 gate).
- Full assemble → `MetEmArtifact.validate(require_optional=True)` passes for d01/d02/d03.

## Predeclared tolerances
S3's interp is graded vs the analytic kernel oracle (≤1e-6 rel). The against-met_em
field tolerances are the schema `parity_tol` values, enforced by S4.

## Validation Commands
```
JAX_PLATFORM_NAME=cpu taskset -c 0-3 python3 -m pytest tests/init/test_interp_metgrid.py \
  tests/init/test_metem_writer.py -q
# optional GPU kernel timing (ONE gpu job at a time):
python3 -m pytest tests/init/test_interp_metgrid.py -q  # with JAX GPU backend
```

## Performance Metrics
Interp is the hot kernel: record wall-clock + (if GPU) a one-shot device-transfer
audit (no host/device bounce inside the per-field loop). Not a v0.3.0 gate, but
recorded — the eventual v0.6.0 operational path needs this on-GPU.

## Proof Object
`proofs/v030/s3_interp_report.json`: per-kernel analytic-oracle error, the
interp_module.F file:line ↔ JAX-function map, the PRES-build check, a structural
diff of the writer output vs a real met_em file, and the full-assemble validate
result.

## Risks
- `four_pt`/`sixteen_pt` index conventions (1-based Fortran, x↔i) — off-by-one and
  i/j transposition are the classic interp bugs; the analytic oracle catches them.
- `search_extrap` fills masked cells from the nearest valid donor — for fine Canary
  targets it is mostly inactive, but the masked-water soil fields exercise it near
  coasts (Tenerife). Validate at a coastal point.
- For fine targets (1/3/9 km ≪ 27 km source) the `_average` coarsening branches are
  inactive; verify the dispatcher does NOT erroneously average-down.

## Handoff Requirements
objective, files, commands, proof, the kernel↔Fortran map, the writer's exact attr
set, unresolved risks, the `MetEmArtifact` assembly entry-point S5 calls.
