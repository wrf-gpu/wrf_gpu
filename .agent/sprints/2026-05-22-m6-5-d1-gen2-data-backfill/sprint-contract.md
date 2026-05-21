# Sprint Contract — M6.5-D1 Gen2 Data Backfill + Quality Audit

**Sprint ID**: `2026-05-22-m6-5-d1-gen2-data-backfill`
**Created**: 2026-05-22 ~23:10 (parallel dispatch with M6.x dycore + F-5 baseline + opus contingency)
**Status**: ACTIVE
**Trigger**: M6-S7 reviewer named M6.5-D1 as the M7 prereq — production Tier-4 RMSE needs a clean Gen2 d02 wrfout corpus accessible from the GPU pipeline. Defer-to-M7 deal struck in M6-S7; this is the prereq sprint that unblocks it.
**Parallel with**: M6.x dycore (file-disjoint: this owns `io/data_*` + `validation/data_quality.py`; M6.x owns dynamics/).

## Objective

Build a production-grade Gen2 d02 wrfout corpus accessor + data-quality audit, sufficient for M7-S0 Tier-4 RMSE harness to consume historical operational forecasts without per-sprint data-acquisition busywork.

## Acceptance

- **AC1 Gen2 d02 wrfout inventory**: scan `/mnt/data/canairy_meteo/runs/wrf_l3/**` (READ-ONLY); produce `artifacts/m6_5/gen2_d02_inventory.json` schema-validated. Include per-run: start_date, hours, d02 wrfout file count, total bytes, complete-or-partial flag, init time, valid-time range.
- **AC2 Data-quality audit**: for each complete run, sample U10/V10/T2/Q2/PSFC/RAINNC fields. Compute: NaN/Inf count, value-range histograms, missing-time-step count, suspicious-spike detector (z-score >5σ flag). Per-run pass/partial/fail status.
- **AC3 GPU-side data loader**: `src/gpuwrf/io/gen2_wrfout_loader.py` — JAX-compatible loader returning per-field arrays + valid_time array. Lazy load (file open per chunk; no full-corpus materialization).
- **AC4 Boundary-replay → wrfout cross-check**: existing `src/gpuwrf/io/boundary_replay.py` reads d02 replay zarr (interpolated from d01 wrfout). Sanity check: compare boundary-replay zarr boundary strip with wrfout boundary strip at lead=0 for same valid-time. If diff > 1% at lead=0, flag as data pipeline bug.
- **AC5 Tier-4 RMSE adapter**: stub function `src/gpuwrf/validation/data_quality.py::compute_rmse_against_gen2(gpu_forecast_state, gen2_wrfout_path, valid_time, fields=['U10','V10','T2'])` that M7-S0 will call. Returns per-field RMSE + per-grid-cell error map.
- **AC6 Selectable Gen2 corpus subset**: support `--start YYYYMMDD --end YYYYMMDD --min-hours N` filtering. Produce subset manifest as `artifacts/m6_5/gen2_d02_subset_<tag>.json` for downstream consumption.
- **AC7 Test surface**: `tests/test_m6_5_gen2_loader.py`, `tests/test_m6_5_data_quality.py`, `tests/test_m6_5_rmse_adapter.py`. Each ≥5 tests including round-trip + edge cases (partial run, missing field, NaN injection).
- **AC8 ADR-016 NEW**: `.agent/decisions/ADR-016-gen2-data-corpus.md` — corpus structure, quality bar, lazy-load policy, M7 RMSE schema.

## Files Worker May Modify

- `src/gpuwrf/io/gen2_wrfout_loader.py` (NEW)
- `src/gpuwrf/io/data_inventory.py` (NEW)
- `src/gpuwrf/validation/data_quality.py` (NEW)
- `scripts/m6_5_gen2_inventory.py` (NEW)
- `scripts/m6_5_data_quality_audit.py` (NEW)
- `tests/test_m6_5_*.py` (NEW)
- `.agent/decisions/ADR-016-gen2-data-corpus.md` (NEW)
- `artifacts/m6_5/**` (NEW, gitignored — files local-only)

## Files Worker Must NOT Modify

- `src/gpuwrf/dynamics/**` (M6.x ownership)
- `src/gpuwrf/coupling/driver.py` (M6.x ownership)
- `src/gpuwrf/contracts/state.py` (M6.x ownership)
- `src/gpuwrf/io/boundary_replay.py` — read only; if extension needed, file an exception in worker report
- `src/gpuwrf/physics/**` (frozen)
- `.agent/decisions/ADR-007-precision-policy.md` (M6.x ownership; F-5 sprint produces a separate denominator amendment ADR)

## Dispatch

- Worker: codex gpt-5.5 xhigh
- Reviewer: Claude Opus 4.7 xhigh (MANDATORY)
- Wall-time: **8-16h**
- Worktree: `/tmp/wrf_gpu2_m65d1`
- Branch: `worker/codex/m65-d1-gen2-data-backfill`

## HARD RULES

1. **/mnt/data/canairy_meteo/** is READ-ONLY. Never write there.
2. NO mocked Gen2 data in tests — use real (small subset) wrfout fixtures or fabricated NetCDF with documented WRF-standard structure.
3. Lazy load. Never materialize full corpus into memory.
4. Tests must be reproducible without /mnt/data access (use fabricated NetCDF fixtures).
5. `/exit` slash-command. Watchdog + multi-Enter.

## End-goal context

This unblocks M7-S0 (Tier-4 RMSE harness). Without it, M7 cannot start. Critical parallel path alongside M6.x dycore.
