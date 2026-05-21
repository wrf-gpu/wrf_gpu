# M6.5-D1 Worker Report - Gen2 Data Backfill + Quality Audit

## Objective

Build the Gen2 d02 `wrfout` corpus accessor, inventory, quality audit,
boundary-replay cross-check, and M7 RMSE adapter required by sprint
`2026-05-22-m6-5-d1-gen2-data-backfill`.

The work followed the repository read order: `PROJECT_CONSTITUTION.md`,
`AGENTS.md`, `.agent/rules/sprint-lifecycle.md`, the sprint contract,
the worker prompt, the M6-S7 reviewer report, `boundary_replay.py`,
`proof_schemas.py`, and the M7 plan under the actual filename
`.agent/sprints/2026-05-21-m7-milestone-plan-scout/m7-milestone-plan.md`.
The prompt's `plan.md` path was absent.

## Files Changed

- `src/gpuwrf/io/data_inventory.py`
- `src/gpuwrf/io/gen2_wrfout_loader.py`
- `src/gpuwrf/validation/data_quality.py`
- `scripts/m6_5_gen2_inventory.py`
- `scripts/m6_5_data_quality_audit.py`
- `tests/test_m6_5_gen2_loader.py`
- `tests/test_m6_5_data_quality.py`
- `tests/test_m6_5_rmse_adapter.py`
- `.agent/decisions/ADR-016-gen2-data-corpus.md`
- `.agent/sprints/2026-05-22-m6-5-d1-gen2-data-backfill/worker-report.md`

Local-only proof artifacts were produced under ignored
`artifacts/m6_5/`.

## AC Results

### AC1 - Gen2 d02 wrfout inventory

Implemented `build_gen2_d02_inventory()` and the CLI
`scripts/m6_5_gen2_inventory.py`. The inventory scans
`/mnt/data/canairy_meteo/runs/wrf_l3` read-only, includes run
directories even when retained d02 history is absent, parses
`wrfout_d02_YYYY-MM-DD_HH:MM:SS` valid times, opens one file per run for
NetCDF dimensions and variables, and validates the output as
`Gen2D02Inventory`.

Measured live archive counts:

- `find /mnt/data/canairy_meteo/runs/wrf_l3 -maxdepth 2 -name "wrfbdy_d01" | wc -l` -> 25
- `find /mnt/data/canairy_meteo/runs/wrf_l3 -maxdepth 2 -name "wrfout_d02_*" | wc -l` -> 78
- inventory `run_count`: 25
- inventory `wrfbdy_d01_run_marker_count`: 25
- inventory `wrfout_d02_file_count`: 78
- complete runs: 3
- partial runs: 22
- inventory `total_bytes`: 1,128,061,809

Complete runs found:

- `20260430_18z_l3_24h_20260520T191306Z`, 25 files, grid 66 x 159
- `20260509_18z_l3_24h_20260511T190519Z`, 25 files, grid 66 x 120
- `20260520_18z_l3_24h_20260521T045847Z`, 25 files, grid 66 x 159

Proof object:
`artifacts/m6_5/gen2_d02_inventory.json`.

### AC2 - Data-quality audit

Implemented `build_quality_audit()` and
`scripts/m6_5_data_quality_audit.py`. The audit opens one wrfout file at
a time and samples `U10`, `V10`, `T2`, `Q2`, `PSFC`, and `RAINNC`.
Metrics include NaN count, Inf count, finite count, min/max, mean/std,
1st/99th percentile histogram, missing timestep count from inventory,
and z-score spike counts using a 5-standard-deviation threshold.

Measured audit summary:

- run records: 25
- sampled complete runs: 3
- status counts: `GREEN=0`, `PARTIAL=25`, `FAIL=0`
- all 22 incomplete runs are `PARTIAL` and unsampled by design
- all 3 complete runs are `PARTIAL` because the spike detector flagged
  surface-field outliers; no NaN/Inf failures were found

Proof object:
`artifacts/m6_5/gen2_d02_quality_audit.json`.

### AC3 - GPU-side data loader

Implemented `src/gpuwrf/io/gen2_wrfout_loader.py`. `Gen2WrfoutLoader`
selects a d02 file by valid time, opens the NetCDF file for that call,
returns per-field arrays plus a `valid_time` array and source file, and
supports `iter_chunks()` for one-file-at-a-time processing. Arrays are
NumPy by default; validation consumers can request JAX arrays with
`as_jax=True` at the consumer boundary.

Tests cover valid-time selection, time-axis round trip, JAX conversion,
missing fields, chunk iteration, and fallback to the WRF `Times`
variable.

### AC4 - Boundary replay to wrfout cross-check

Implemented `compare_boundary_replay_to_wrfout()` in the new validation
module without modifying `src/gpuwrf/io/boundary_replay.py`. The check
compares lead-0 zarr boundary strips to d02 wrfout boundary strips at
the same valid time and fails when relative MAE exceeds 1 percent.

Measured cross-check:

- replay: `data/fixtures/m6/d02_boundary_replay_v2.zarr`
- run: `/mnt/data/canairy_meteo/runs/wrf_l3/20260520_18z_l3_24h_20260521T045847Z`
- valid time: `2026-05-20T18:00:00+00:00`
- status: `FAIL`
- failure: `V/E rel_mae=0.010208 > 0.01`

This is recorded as a data-pipeline flag under the sprint rule. Other
variable aggregate max relative MAEs were below the threshold.

Proof object:
`artifacts/m6_5/gen2_boundary_replay_cross_check.json`, also embedded
inside the quality audit artifact.

### AC5 - Tier-4 RMSE adapter

Implemented
`compute_rmse_against_gen2(gpu_forecast_state, gen2_wrfout_path,
valid_time, fields=("U10","V10","T2"))`. It accepts a Gen2 run
directory or single wrfout file, validates the requested valid time for
file inputs, converts truth and forecast fields to JAX arrays at the
adapter boundary, and returns per-field `rmse`, `error_map`,
`valid_time_utc`, and `gen2_source_file`.

Tests cover zero RMSE, nonzero per-cell error maps, run-directory valid
time selection, missing forecast fields, shape mismatch, and file
valid-time mismatch.

### AC6 - Selectable Gen2 corpus subset

Implemented subset filtering by `--start YYYYMMDD --end YYYYMMDD
--min-hours N` in `scripts/m6_5_gen2_inventory.py`. The subset defaults
to complete runs only unless `--include-partial` is set.

Produced subset:

- command tag: `complete24h`
- output: `artifacts/m6_5/gen2_d02_subset_complete24h.json`
- filters: start `20260401`, end `20260531`, min-hours `24`,
  require-complete `true`
- subset runs: 3
- subset files: 75
- subset bytes: 1,086,421,525

### AC7 - Test surface

Added three required test files using fabricated WRF-standard NetCDF
fixtures created under `tmp_path`; no test requires `/mnt/data`.

- `tests/test_m6_5_gen2_loader.py`: 6 tests
- `tests/test_m6_5_data_quality.py`: 7 tests
- `tests/test_m6_5_rmse_adapter.py`: 6 tests

Validation run:

- `pytest -q tests/test_m6_5_gen2_loader.py tests/test_m6_5_data_quality.py tests/test_m6_5_rmse_adapter.py` -> 19 passed
- `pytest -q tests/test_m6_5_gen2_loader.py tests/test_m6_5_data_quality.py tests/test_m6_5_rmse_adapter.py tests/test_m6_gen2_accessor.py tests/test_m6_boundary_replay.py` -> 25 passed
- `python -m compileall -q src/gpuwrf/io/data_inventory.py src/gpuwrf/io/gen2_wrfout_loader.py src/gpuwrf/validation/data_quality.py scripts/m6_5_gen2_inventory.py scripts/m6_5_data_quality_audit.py` -> pass
- `python -m json.tool` on all four `artifacts/m6_5/*.json` proof objects -> pass

### AC8 - ADR-016

Added `.agent/decisions/ADR-016-gen2-data-corpus.md`. It records the
Gen2 corpus path, file naming convention, complete-run definition,
quality status bar, lazy-load policy, boundary cross-check policy, and
M7 RMSE adapter schema.

## Commands Run

- `sed -n '1,240p' PROJECT_CONSTITUTION.md`
- `sed -n '1,260p' AGENTS.md`
- `sed -n '1,260p' .agent/rules/sprint-lifecycle.md`
- `sed -n '1,280p' .agent/sprints/2026-05-22-m6-5-d1-gen2-data-backfill/sprint-contract.md`
- `sed -n '1,320p' .agent/sprints/2026-05-22-m6-5-d1-gen2-data-backfill/role-prompts/worker.md`
- `sed -n '1,260p' .agent/sprints/2026-05-21-m6-s7-tier4-probtest/reviewer-report.md`
- `sed -n '1,320p' src/gpuwrf/io/boundary_replay.py`
- `sed -n '1,340p' src/gpuwrf/io/proof_schemas.py`
- `sed -n '1,320p' .agent/sprints/2026-05-21-m7-milestone-plan-scout/m7-milestone-plan.md`
- `find /mnt/data/canairy_meteo/runs/wrf_l3 -maxdepth 2 -name 'wrfbdy_d01' | wc -l`
- `find /mnt/data/canairy_meteo/runs/wrf_l3 -maxdepth 2 -name 'wrfout_d02_*' | wc -l`
- `python scripts/m6_5_gen2_inventory.py --start 20260401 --end 20260531 --min-hours 24 --tag complete24h`
- `python scripts/m6_5_data_quality_audit.py --boundary-replay data/fixtures/m6/d02_boundary_replay_v2.zarr --boundary-run /mnt/data/canairy_meteo/runs/wrf_l3/20260520_18z_l3_24h_20260521T045847Z`
- test and validation commands listed under AC7

## Proof Objects Produced

- `artifacts/m6_5/gen2_d02_inventory.json`
- `artifacts/m6_5/gen2_d02_quality_audit.json`
- `artifacts/m6_5/gen2_d02_subset_complete24h.json`
- `artifacts/m6_5/gen2_boundary_replay_cross_check.json`
- `.agent/decisions/ADR-016-gen2-data-corpus.md`
- this worker report

## Unresolved Risks

- AC4 is not green. The v2 boundary replay differs from d02 wrfout at
  lead 0 on `V/E` by 1.0208 percent relative MAE, just above the sprint
  threshold. This should be reviewed as a data-pipeline issue before M7
  relies on that replay fixture.
- The live corpus still has only 3 complete d02 24-hour runs, and one of
  them is the older 66 x 120 grid. This inventory accessor unblocks M7
  code paths, but it does not create the 10-member pinned-grid corpus
  that Tier-4 production tolerances ultimately need.
- The spike detector is intentionally simple and flags all complete runs
  as `PARTIAL`. That is conservative for data QA, but a reviewer may
  choose to tune field-specific spike policy in a later contract.
- Independent Opus review remains mandatory before manager close because
  this sprint adds production code and ADR-016.

## Next Decision Needed

Dispatch the required independent Opus review. The reviewer should
decide whether the AC4 `V/E` boundary-replay failure is an accepted
known fixture limitation, a bug requiring a follow-up fix before M7-S0,
or evidence that the 1 percent cross-check threshold needs a separately
approved contract amendment.
