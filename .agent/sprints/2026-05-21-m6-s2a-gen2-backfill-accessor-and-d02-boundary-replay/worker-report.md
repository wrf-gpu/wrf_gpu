# M6-S2a Worker Report — Gen2 Accessor, d02 Boundary Replay, Shared I/O

## Objective

Build the shared M6 I/O layer required by M6-S2 and M6-S4..S8: read-only Gen2
backfill access, d02 lateral-boundary replay from parent d01 history, shared
validation loaders/regridding/masks/unit conversion, fair CPU denominator
extraction, M6 proof-object schemas, and ADR-011.

## Outcome

Implemented the contracted infrastructure and generated the proof objects. The
pinned Gen2 run was treated as read-only throughout. The read-only exploration
confirmed `wrfbdy_d02` is absent and that hourly `wrfout_d01_*` and
`wrfout_d02_*` files exist for 25 output times from 2026-05-19 18:00 UTC through
2026-05-20 18:00 UTC.

## Acceptance Criteria

AC1 — Gen2 backfill accessor: PASS.
`src/gpuwrf/io/gen2_accessor.py` adds `Gen2Run`, lazy variable handles, namelist
parsing, variable inventory, domain discovery, and `Gen2GridSpec` metadata with
`as_grid_spec()` conversion. `artifacts/m6/gen2_manifest.json` was generated with
5 domains, 133 files, SHA-256 hashes, mtimes, sizes, variable inventories, and
`no_write_audit: true`. d02 metadata matches the WRF namelist and first wrfout:
dx/dy 3000 m, `e_we=160`, `e_sn=67`, `e_vert=45`, mass grid `159 x 66 x 44`,
Lambert projection, parent d01, parent ratio 3.

AC2 — d02 boundary replay: PASS.
`src/gpuwrf/io/boundary_replay.py` writes
`data/fixtures/m6/d02_boundary_replay_v1.zarr` and
`fixtures/manifests/m6_d02_boundary_replay.yaml`. It extracts U, V, T, QVAPOR,
and PH per side W/E/S/N for each hourly d02 history time. Horizontal interpolation
is bilinear in native WRF Lambert coordinates using the variable's own stagger
coordinates (`XLAT/XLONG`, `XLAT_U/XLONG_U`, or `XLAT_V/XLONG_V`). Vertical
interpolation is linear in eta; for this pinned d01/d02 pair the eta arrays are
identical, so the vertical interpolation is an identity under the linear-eta rule.

Round-trip evidence against `wrfout_d02_*` boundary cells passed the documented
tolerances. Aggregate maxima across all sides and times:

- U: RMSE 0.1268 m/s, relative MAE 0.00339, tolerance 0.5 m/s and 3%.
- V: RMSE 0.1831 m/s, relative MAE 0.01060, tolerance 0.5 m/s and 3%.
- T: RMSE 0.2084 K, tolerance 0.5 K.
- QVAPOR: RMSE 8.36e-05 kg/kg, relative MAE 0.00925, tolerance 1e-4 and 3%.
- PH: RMSE 6.03 m2/s2, relative MAE 0.000672, tolerance 20 m2/s2 and 0.5%.

AC3 — Shared validation I/O: PASS.
`src/gpuwrf/io/validation.py` adds `load_gen2_var`, `regrid`, `domain_mask`,
`lead_time_slice`, and `unit_convert`. M6 validation sprints should import these
helpers directly from `gpuwrf.io.validation`; ADR-011 records this ownership rule.

AC4 — CPU denominator extractor: PASS with a precision caveat.
`scripts/m6_extract_cpu_denominator.py` parses `rsl.error.0000`, namelist grid
metadata, and the Gen2 WRF compile log to write `artifacts/m6/cpu_denominator.json`.
The artifact reports total d01 nested-run timing sum 17010.36973 s, d02
attribution fraction 0.1826091496, d02-attributable 24h wall time
3106.249150758174 s, and d02-attributable per-step time 215.71174658042875 ms.
The attribution policy is grid-points times timestep-count fraction, with no cap
or `min(raw, cap)` adjustment.

The compile log records NVHPC 26.3 and flags including `-O3`, `-acc`,
`-gpu=cc120,fastmath`, `-Mfree`, `-byteswapio`, `-Mrecursive`, `-r4`, and `-i4`.
The sprint prompt example requested `fp_precision: FP64`, but the observed compile
evidence includes `-r4` default real. I recorded that honestly in the artifact
instead of overwriting the evidence. M6-S5 should decide how to interpret the
precision mismatch before using the denominator as a binding 4x verdict input.

AC5 — Proof-object schemas: PASS.
`src/gpuwrf/io/proof_schemas.py` defines machine-readable JSON schemas and
validators for `CoupledDummyCarry`, `SpacetimeBudget`, `ForecastSmoke`,
`Forecast24h`, `Tier2CoupledInvariants`, `Tier3DriftEnvelope`,
`Tier4ProbtestTolerances`, `Gen2Comparison`, `FullDomainBatchingVerdict`, and
`MilestoneCloseoutM6`. Existing committed M6-S1 artifacts
`artifacts/m6/coupled_dummy_carry.json` and `artifacts/m6/spacetime_budget.json`
validate through the registry.

AC6 — ADR-011: PASS.
`.agent/decisions/ADR-011-m6-shared-io-and-boundary-replay.md` documents the
shared I/O owner, Gen2 read-only contract, d02 replay strategy, proof-object
registry, fair CPU denominator policy, and cross-references ADR-002, ADR-007,
ADR-010, the CPU WRF baseline reference, the pinned WRF namelist, and the Gen2
WRF reference.

AC7 — Honest accounting: PASS with noted caveat.
The read-only audit command returns `READ-ONLY OK`. The Gen2 accessor uses
read-mode NetCDF opens and lazy device caching after first materialization.
No post-init timestep-loop transfers are introduced by this sprint; the accessor
performs explicit load-time materialization only. No `min(raw, cap)` or equivalent
speedup denominator cap was used.

## Files Changed

- `src/gpuwrf/io/__init__.py`
- `src/gpuwrf/io/gen2_accessor.py`
- `src/gpuwrf/io/boundary_replay.py`
- `src/gpuwrf/io/validation.py`
- `src/gpuwrf/io/proof_schemas.py`
- `scripts/m6_extract_cpu_denominator.py`
- `tests/test_m6_gen2_accessor.py`
- `tests/test_m6_boundary_replay.py`
- `tests/test_m6_validation_io.py`
- `tests/test_m6_proof_schemas.py`
- `.agent/decisions/ADR-011-m6-shared-io-and-boundary-replay.md`
- `artifacts/m6/gen2_manifest.json`
- `artifacts/m6/cpu_denominator.json`
- `data/fixtures/m6/d02_boundary_replay_v1.zarr`
- `fixtures/manifests/m6_d02_boundary_replay.yaml`

## Commands Run

- `sed -n '1,240p' PROJECT_CONSTITUTION.md`
- `sed -n '1,260p' AGENTS.md`
- `sed -n '1,320p' .agent/rules/sprint-lifecycle.md`
- `sed -n '1,360p' .agent/sprints/2026-05-21-m6-s2a-gen2-backfill-accessor-and-d02-boundary-replay/sprint-contract.md`
- Read project-local skills `validating-physics`, `building-wrf-oracles`, and `reporting-to-human`.
- Read M6 critic amendments, manager amendments, ADR-010, CPU baseline, Gen2 baseline memory, validation philosophy memory, and dispatching pattern.
- `ls`, `head`, and `ls wrfout/wrfbdy` probes on the pinned Gen2 run, read-only.
- NetCDF metadata probes for domain shapes, projection attributes, variables, and boundary replay prototype errors.
- `python -m py_compile src/gpuwrf/io/__init__.py src/gpuwrf/io/gen2_accessor.py src/gpuwrf/io/validation.py src/gpuwrf/io/boundary_replay.py src/gpuwrf/io/proof_schemas.py scripts/m6_extract_cpu_denominator.py`
- `python -m pip install -e .` so the sprint contract's plain `python -c` import works in this shell.
- `python scripts/m6_extract_cpu_denominator.py`
- `python -c "from gpuwrf.io.gen2_accessor import Gen2Run; r = Gen2Run('/mnt/data/canairy_meteo/runs/wrf_l3/20260519_18z_l3_24h_20260520T025228Z'); print(r.domains); print(r.grid('d02'))"`
- Generated `artifacts/m6/gen2_manifest.json` with `Gen2Run.write_manifest(..., include_sha256=True)`.
- `python -c "from gpuwrf.io.gen2_accessor import Gen2Run; from gpuwrf.io.boundary_replay import extract_d02_boundary; ..."`
- `pytest -q tests/test_m6_gen2_accessor.py tests/test_m6_boundary_replay.py tests/test_m6_validation_io.py tests/test_m6_proof_schemas.py`

## Proof Objects Produced

- `artifacts/m6/gen2_manifest.json`
- `artifacts/m6/cpu_denominator.json`
- `data/fixtures/m6/d02_boundary_replay_v1.zarr`
- `data/fixtures/m6/d02_boundary_replay_v1.zarr/validation_summary.json`
- `fixtures/manifests/m6_d02_boundary_replay.yaml`
- Focused pytest result: 10 passed in 2.57 s.

## Unresolved Risks

- CPU denominator precision evidence conflicts with the prompt's example
  `FP64` field. The compile log evidence says `-r4`; I preserved the evidence and
  flagged it for M6-S5.
- Boundary replay is physically consistent but not WRF `wrfbdy_d02` bitwise
  reconstruction. It is suitable for M6-S2 forcing because tolerances are declared
  before use and validated against d02 history.
- `zarr` is available in this environment, but `pyproject.toml` did not previously
  declare it. I did not edit `pyproject.toml` because it was outside the contract's
  allowed file set.

## Next Decision Needed

Mandatory Claude Opus 4.7 reviewer pass per sprint lifecycle. Reviewer should
specifically check the CPU denominator precision caveat, the boundary replay
interpolation assumptions, and whether M6-S5 wants an explicit follow-up on the
`-r4` versus `FP64` mismatch before using the denominator as binding evidence.
