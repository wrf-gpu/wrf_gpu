# Sprint Contract — M6-S2a Gen2 Backfill Accessor + d02 Boundary Replay + Shared Validation I/O

**Sprint ID**: `2026-05-21-m6-s2a-gen2-backfill-accessor-and-d02-boundary-replay`
**Created**: 2026-05-21 11:42 by manager (Claude Opus 4.7 1M-context)
**Status**: STUB — to dispatch after M6-S1 Opus closes (parallel with M5-S3.z, file-disjoint)
**Trigger**: Codex M6-plan critic amendment #2: "Add S2a or equivalent: Gen2 backfill accessor, read-only manifest, d02 boundary replay/regridding, output schema, and CPU denominator extractor. Without this, standalone d02 24h forecast is impossible because no `wrfbdy_d02` exists in the Gen2 nested run."

## Objective

Build the **shared I/O + validation infrastructure** layer that M6-S2 forecast driver + M6-S4..S8 validation sprints all depend on. Critic amendment #2 (load-bearing) + #5 (proof-object schemas) + #6 (one sprint owns shared validation I/O) combined here.

Five deliverables:

1. **Gen2 backfill accessor** — read-only adapter for `/mnt/data/canairy_meteo/runs/wrf_l3/20260519_18z_l3_24h_20260520T025228Z/` with manifest (path, SHA, mtime, variables, domain ID, no-write audit).
2. **d02 boundary replay** — extract lateral BCs for the 3 km `d02` domain from the nested parent (`wrfout_d01` interpolated to d02 grid). Since no `wrfbdy_d02` exists, we synthesize it from parent-domain history.
3. **Shared validation I/O** — single owner for: NetCDF/zarr loaders, domain masks, regridding (parent-to-child, child-to-parent, observation-to-model-grid), variable name maps, unit conversions, lead-time selection.
4. **CPU denominator extractor** — for ADR-007 4× verdict (M6-S5): isolate the d02-only CPU wall-time from the 5-domain nested Gen2 run, with metadata (which CPU, which compile flags, parent-grid contribution exclusion policy).
5. **Output manifest schema** — JSON schema for all M6 proof objects (`coupled_dummy_carry.json`, `forecast_6h_summary.json`, `gen2_comparison.json`, `tier2_coupled_invariants.json`, etc.) so M6-S2..S8 sprints all use the same shape.

## Acceptance

- **AC1 Gen2 accessor**. `src/gpuwrf/io/gen2_accessor.py` provides `Gen2Run` class with read-only API. Manifest at `artifacts/m6/gen2_manifest.json` with: run-ID, full path, file list with SHAs/mtimes, domain IDs (d01..d05), grid metadata per domain, variable inventory, "no-write audit" boolean.
- **AC2 d02 boundary replay**. `src/gpuwrf/io/boundary_replay.py` extracts d02 lateral BCs from `wrfout_d01_*` files via parent-to-child regridding. Output: `data/fixtures/m6/d02_boundary_replay_v1.{zarr,npz}` with per-side per-variable per-time BCs. Validation: round-trip a sample d02 interior cell from `wrfout_d02_*` against the replayed BC to confirm physical consistency.
- **AC3 shared validation I/O**. `src/gpuwrf/io/validation.py` provides: `load_gen2_var(run, domain, var, time)`, `regrid(src_field, src_grid, dst_grid, method)`, `domain_mask(grid, region)`, `lead_time_slice(run, lead_hours)`, `unit_convert(field, from_unit, to_unit)`. Hard rule: every M6-S4..S8 sprint imports from this module, not from anywhere else.
- **AC4 CPU denominator extractor**. `scripts/m6_extract_cpu_denominator.py` parses Gen2 run logs + namelists to produce `artifacts/m6/cpu_denominator.json` with: wall-time-per-step, total wall-time, domain-only-attributable wall-time (d02 fraction), compile flags, hardware ID, NVHPC version, attribution policy.
- **AC5 output manifest schema**. `src/gpuwrf/io/proof_schemas.py` defines JSON schemas (pydantic or similar) for: coupled_dummy_carry, forecast_6h_summary, tier2_coupled_invariants, tier3_drift_envelope, tier4_probtest_tolerances, gen2_comparison, full_domain_batching_verdict, milestone_closeout. Tests assert every M6 schema validates.
- **AC6 ADR-011**. `.agent/decisions/ADR-011-m6-shared-io-and-boundary-replay.md` documenting the shared-I/O owner, the boundary-replay strategy, and the proof-object schema registry. Cross-reference ADR-002 (state), ADR-007 (precision), ADR-010 (M6 state extension).
- **AC7 Honest accounting**. Read-only audit on Gen2 paths (no writes to `/mnt/data/canairy_meteo/`). Zero post-init transfers in the load path (lazy + device-resident after first read).

## Files Worker May Modify

- `src/gpuwrf/io/__init__.py`, `gen2_accessor.py`, `boundary_replay.py`, `validation.py`, `proof_schemas.py` (NEW module)
- `scripts/m6_extract_cpu_denominator.py` (NEW)
- `data/fixtures/m6/d02_boundary_replay_v1.{zarr,npz}` (NEW)
- `fixtures/manifests/m6_d02_boundary_replay.yaml` (NEW)
- `tests/test_m6_gen2_accessor.py`, `test_m6_boundary_replay.py`, `test_m6_validation_io.py`, `test_m6_proof_schemas.py` (NEW)
- `.agent/decisions/ADR-011-m6-shared-io-and-boundary-replay.md` (NEW)
- `artifacts/m6/gen2_manifest.json`, `cpu_denominator.json` (NEW)
- Worker report

## Files Worker Must NOT Modify

- `src/gpuwrf/contracts/**`, `coupling/**`, `dynamics/**`, `physics/**`, `timestep/**`
- ANY file under `/mnt/data/canairy_meteo/` (READ-ONLY — hard rule per project memory `project_canairy_meteo_baseline.md`)
- Other ADR or governance file

## Dispatch

- Primary worker: codex gpt-5.5 xhigh
- Reviewer: Claude Opus 4.7 xhigh (mandatory per sprint-lifecycle)
- Wall-time: **12-18 hours**
- Worktree: `/tmp/wrf_gpu2_m6s2a` (NEW, isolated)
- Branch: `worker/codex/m6-s2a-gen2-backfill-accessor-and-d02-boundary-replay`

## Hard rules

- **READ-ONLY** on `/mnt/data/canairy_meteo/**` — absolute. Any write attempt is a P0 blocker.
- Boundary replay must be physically consistent (validate via round-trip test, not just "data loaded").
- Proof-object schemas must be machine-readable AND human-checkable (pydantic + docstring).
- Cite WRF namelist + Gen2 README for every domain/grid metadata claim.
- M6-S2 forecast driver, M6-S4..S8 validation, and M6-S5 4× verdict ALL depend on this; treat as blocking infrastructure not an optional accessory.

## Sequencing impact

- M6-S2 (full d02 forecast) BLOCKED on this close (boundary replay required).
- M6-S5 4× verdict BLOCKED on this close (fair CPU denominator required).
- M6-S4/S6/S7/S8 BLOCKED on this close (shared validation I/O required).
- M6-S3 surface/Noah-MP can START in parallel (it consumes Gen2 surface data, but the accessor API gives it that).
