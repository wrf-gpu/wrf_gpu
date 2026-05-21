# ADR-011 — M6 Shared I/O And d02 Boundary Replay

Date: 2026-05-21
Author: M6-S2a worker (codex gpt-5.5 xhigh)
Status: PROPOSED for M6-S2a reviewer
Scope: `src/gpuwrf/io/**`, Gen2 read-only backfill access, d02 lateral-boundary replay fixture, M6 proof-object schema registry, and fair CPU denominator extraction.

## Context

M6 pins the Gen2 backfill run
`/mnt/data/canairy_meteo/runs/wrf_l3/20260519_18z_l3_24h_20260520T025228Z`.
The run has `wrfout_d01_*` through `wrfout_d05_*`, `wrfinput_d01..d05`, and
`wrfbdy_d01`; it does not have `wrfbdy_d02`. The run `namelist.input` records
`max_dom = 5`, `dx = 9000, 3000, 1000, 1000, 1000`, `e_we = 94, 160, 94, 70, 70`,
`e_sn = 60, 67, 76, 61, 58`, `e_vert = 45`, and d02 nesting under d01 with
`parent_grid_ratio = 3`, `i_parent_start = 24`, `j_parent_start = 20`.

The project Gen2 reference says these backfill outputs are the authoritative
CPU WRF validation source and that this repository must never write under
`/mnt/data/canairy_meteo/**`. The Gen2 WRF reference documents the NVHPC 26.3
WRF environment and WRF 4.7.1 source layout used by the existing operational
pipeline. This ADR makes that read-only data source usable by M6-S2 and the
validation sprints without duplicating loader and regridding logic.

## Decision

`src/gpuwrf/io/**` is the shared M6 I/O owner. M6-S2, M6-S4, M6-S5, M6-S6,
M6-S7, and M6-S8 must import Gen2 loading, unit conversion, regridding,
domain masks, lead-time selection, and proof-schema validation from this package
instead of creating sprint-local variants.

The Gen2 access contract is read-only:

- `Gen2Run` opens NetCDF sources in read mode only.
- `Gen2Run.load(..., lazy=True)` returns a lazy handle; materialization caches a
  JAX device array so repeated reads do not create repeated host/device copies.
- `Gen2Run.write_manifest(...)` rejects output targets inside the Gen2 data tree.
- Generated project artifacts live under this repository only.

The d02 boundary strategy is replay, not a synthetic closed-boundary case.
`extract_d02_boundary(Gen2Run, output_path)` reads parent `wrfout_d01_*` history,
interpolates parent fields to d02 lateral boundary coordinates, and writes
`data/fixtures/m6/d02_boundary_replay_v1.zarr`. The frozen variables are
`U`, `V`, `T`, `QVAPOR`, and `PH`, split by side `W/E/S/N` and by hourly Gen2
history time. Horizontal interpolation is bilinear in native WRF Lambert space,
using the field's own mass/U/V stagger coordinates. Vertical interpolation is
linear in eta; for the pinned Gen2 d01/d02 pair the eta arrays are identical, so
the implementation records an identity vertical interpolation under the same
linear-eta rule.

The proof-object schema registry lives in `src/gpuwrf/io/proof_schemas.py`.
It defines machine-readable and human-checkable schemas for M6 artifacts:
`CoupledDummyCarry`, `SpacetimeBudget`, `ForecastSmoke`, `Forecast24h`,
`Tier2CoupledInvariants`, `Tier3DriftEnvelope`, `Tier4ProbtestTolerances`,
`Gen2Comparison`, `FullDomainBatchingVerdict`, and `MilestoneCloseoutM6`.
M6 sprints must validate their JSON proof objects through this registry rather
than emitting ad-hoc JSON.

The fair CPU denominator for ADR-007 is domain-scoped. M6-S5 must use
`artifacts/m6/cpu_denominator.json`, which attributes a fraction of total
five-domain nested WRF wall time to d02 by grid-point count times timestep count.
It must not compare a d02-only GPU run against the full five-domain wall time.

## Consequences

Positive:

- M6-S2 can run a real d02 forecast with replayed lateral forcing instead of a
  closed diagnostic crop.
- M6-S4/S6/S7/S8 share loader, mask, unit, and regridding semantics.
- M6-S5 has a documented denominator policy and raw timing provenance.
- M6 closeout can validate proof objects by schema name.

Risks:

- Boundary replay is not bitwise WRF `wrfbdy` regeneration. It is a physically
  consistent parent-history replay fixture with documented interpolation and
  nested-feedback tolerances.
- The CPU denominator compile log records `-r4` default-real flags in the Gen2
  WRF GPU source tree. M6-S5 must account for that evidence when interpreting
  precision and speedup claims.
- `GridSpec.as_grid_spec()` uses real Gen2 terrain and eta metadata, but it does
  not by itself authorize operational validation; downstream sprints still need
  their own proof objects.

## Cross-References

- ADR-002: SoA state layout and C-grid staggering remain the model-side state
  contract.
- ADR-007: M6-S5 speed claims must use the fair CPU denominator and operational
  RMSE gates.
- ADR-010: M6-S1 froze the coupled state, precision registry, adapter names, and
  downstream file ownership.
- `.agent/references/cpu-wrf-baseline.md`: authoritative Gen2 backfill and
  read-only contract.
- Gen2 WRF reference: `/home/enric/src/canairy_meteo/Gen2/wrf-gpu.md`.
- Pinned WRF namelist: `/mnt/data/canairy_meteo/runs/wrf_l3/20260519_18z_l3_24h_20260520T025228Z/namelist.input`.
