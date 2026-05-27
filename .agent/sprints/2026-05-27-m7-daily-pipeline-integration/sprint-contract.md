# Sprint Contract — M7 Daily-Pipeline Integration (End-to-End)

**Sprint ID**: `2026-05-27-m7-daily-pipeline-integration`
**Created**: 2026-05-27 (autonomous overnight loop — the M7-close sprint)
**Status**: READY
**Predecessors**:
- M7 perf-measurement (`b7d9fe7`) — 5.71s warm wall-clock, D2H=0 invariant
- M7 1km memory audit (`7907d7b`) — FITS_WITH_HEADROOM
- M7 NetCDF writer (`4c20de3`) — WRITER_READY, 0 critical gaps
- M7 restart-continuity (`ec072dd`) — PASS bitwise
- M7 forecast-vs-obs scaffold (`ce04cae`) — SCAFFOLD_READY
- M7 wrfout I/O compat audit (`a181d68`) — compat matrix done

## Objective

This is the M7 close gate. Wire together every component the prior sprints built into a single end-to-end **daily pipeline driver** that:

1. Loads AIFS IC/BC for a given Canary day (via `gpuwrf.integration.d02_replay.build_replay_case` adapted for daily-run inputs)
2. Runs the GPU forecast on the Canary d02 3km domain for N hours (configurable; default 24h)
3. Optionally checkpoints at mid-run via `gpuwrf.runtime.checkpoint.write_checkpoint`
4. Writes each output time as a WRF-compatible NetCDF file via `gpuwrf.io.wrfout_writer.write_wrfout_netcdf`
5. Scores the resulting wrfouts against AEMET station observations via `gpuwrf.validation.forecast_vs_obs`
6. Emits a single daily-run proof object summarizing wall-clock, output files, station scores, and gate verdicts

Sprint definition of done: **one full Canary 3km day driven end-to-end from AIFS IC/BC to scored wrfout, repeatable.** Per M7 acceptance gate #4.

## Acceptance

- **AC1 — Pipeline driver**: implement `scripts/m7_daily_pipeline.py` with CLI:
  ```
  python -m scripts.m7_daily_pipeline \
      --run-id 20260521_18z_l3_24h_20260522T072630Z \
      --hours 24 \
      --output-dir /tmp/m7_pipeline_runs/20260521 \
      --score
  ```
  The driver orchestrates IC load → forecast → write → score. Use `taskset -c 0-3` for the Python orchestration; the JAX kernels use the GPU.

- **AC2 — End-to-end run on 20260521**: execute the driver on the 20260521 V3 IC for **24 hours** (full operational forecast length). Emit `.agent/sprints/2026-05-27-m7-daily-pipeline-integration/pipeline_run_20260521.json` with: wall-clock total, wall-clock per hour, per-hour wrfout filenames, station score summary, all-finite check, verdict.

- **AC3 — Output wrfouts**: write hourly `wrfout_d02_YYYY-MM-DD_HH:00:00` NetCDF files (24 of them for a 24h forecast). Verify each is readable via `netCDF4.Dataset(...)` and contains the 41-var minimum subset (per the NetCDF writer's prior compat_matrix_v2). Emit `.agent/sprints/2026-05-27-m7-daily-pipeline-integration/wrfout_inventory.json`.

- **AC4 — Station scoring against AEMET**: run `gpuwrf.validation.forecast_vs_obs` against the 24 generated wrfouts. Emit `.agent/sprints/2026-05-27-m7-daily-pipeline-integration/station_scores_20260521.json` with: per-variable BIAS / RMSE / MAE / sample count for T2, U10, V10, optionally precip FSS. **This is a measurement, not a gate** — the only acceptance bar is that scores are **finite** and the join produces ≥ 100 station-rows. (Skill claim against CPU baseline is left to a downstream comparison sprint.)

- **AC5 — Restart probe during pipeline**: include a `--restart-at-hour H` option that checkpoints at hour H, then continues. Verify the resulting 24h forecast bitwise matches a continuous run via the restart-continuity machinery from `tests/test_m7_restart_checkpoint_roundtrip.py`. Emit `.agent/sprints/2026-05-27-m7-daily-pipeline-integration/restart_in_pipeline.json` with PASS/FAIL.

- **AC6 — Repeatability**: run the same 20260521 pipeline TWICE end-to-end; compare the two runs' final-hour wrfouts via xarray; verify bitwise (FP64) / Tier-1 tolerance (FP32) equality. Emit `.agent/sprints/2026-05-27-m7-daily-pipeline-integration/repeatability.json`.

- **AC7 — Wall-clock vs CPU baseline**: compare the 24h pipeline wall-clock against the existing Gen2 `wrf_l3/20260521_*/namelist.output` per-step timing × steps-per-24h. Emit `.agent/sprints/2026-05-27-m7-daily-pipeline-integration/speedup_vs_cpu_24h.json`. Target was 4-8×; the 1h preliminary was 100-1900×; 24h should remain in the speedup band but may be lower due to I/O / scoring overhead.

- **AC8 — Worker report** with verdict `PIPELINE_GREEN` / `PIPELINE_PARTIAL` / `PIPELINE_BLOCKED` and the M7 close recommendation.

## Files Worker May Modify

- `scripts/m7_daily_pipeline.py` (NEW — the orchestrator)
- `src/gpuwrf/integration/daily_pipeline.py` (NEW, if a Python module is cleaner than a script)
- `src/gpuwrf/runtime/__init__.py` (export new pipeline entry point if appropriate)
- `tests/test_m7_daily_pipeline.py` (NEW — pipeline unit tests with synthetic state, NOT a full 24h GPU forecast)
- `.agent/sprints/2026-05-27-m7-daily-pipeline-integration/**`

## Files Worker Must Not Modify

- `src/gpuwrf/runtime/operational_mode.py` — pipeline calls into operational mode, does not re-implement it
- `src/gpuwrf/contracts/state.py`, `src/gpuwrf/contracts/precision.py`
- `src/gpuwrf/io/wrfout_writer.py` — writer is frozen at WRITER_READY
- `src/gpuwrf/runtime/checkpoint.py` — checkpoint is frozen
- `src/gpuwrf/validation/forecast_vs_obs.py` — scaffold is frozen
- governance files
- `/mnt/data/canairy_meteo/**` — read-only

## Hard Rules

1. **No re-implementation.** Pipeline is composition over existing modules.
2. **GPU**: uses RTX 5090 freely. Nightly WRF is on CPU cores 4-31 — do not interfere.
3. **CPU pinning**: `taskset -c 0-3` for the Python orchestrator.
4. **Output discipline**: pipeline outputs go to `/tmp/m7_pipeline_runs/` (large, not committed); JSON proof artifacts go to `.agent/sprints/2026-05-27-m7-daily-pipeline-integration/`.
5. **No remote push.** Local commit on `worker/gpt/m7-daily-pipeline-integration` only.
6. **Honest BLOCKED**: if 24h forecast doesn't fit memory / OOMs / develops nonfinite physics, emit PIPELINE_BLOCKED with the explicit failure mode. Do not silently lengthen or shorten the forecast window.
7. **Read-only on AEMET data** under `/mnt/data/canairy_meteo/artifacts/datasets/aemet_stations/`.

## Dependencies

- M7 perf-measurement step + all 5 prior sprints (writer, checkpoint, scaffold, audit, 1km)
- RTX 5090 + 32 GB VRAM (1km audit confirmed headroom; 3km uses ~7 GB)
- Gen2 backfill for 20260521 + AIFS month files for 2026-05

## Proof Objects

- `.agent/sprints/2026-05-27-m7-daily-pipeline-integration/pipeline_run_20260521.json` (AC2 — main gate)
- `.agent/sprints/2026-05-27-m7-daily-pipeline-integration/wrfout_inventory.json` (AC3)
- `.agent/sprints/2026-05-27-m7-daily-pipeline-integration/station_scores_20260521.json` (AC4)
- `.agent/sprints/2026-05-27-m7-daily-pipeline-integration/restart_in_pipeline.json` (AC5)
- `.agent/sprints/2026-05-27-m7-daily-pipeline-integration/repeatability.json` (AC6)
- `.agent/sprints/2026-05-27-m7-daily-pipeline-integration/speedup_vs_cpu_24h.json` (AC7)
- `.agent/sprints/2026-05-27-m7-daily-pipeline-integration/worker-report.md` (AC8)

## Dispatch

- Worker: codex gpt-5.5 xhigh
- Wall-time: 6-12 h (this is the big one)
- Branch: `worker/gpt/m7-daily-pipeline-integration`
- Worktree: `/tmp/wrf_gpu2_pipeline`
- GPU usage: YES (24h × 2 forecasts + restart probe)

## What this closes

If PIPELINE_GREEN: M7 gates #4 closed, gates #2 (writer integration) and #6 (scaffold integration) elevated from "scaffold ready" to "wired in", and the M7 close becomes feasible (with the two corpus-blocked gates #1 + #7 escalated to user direction or covered by the parallel Gen2 corpus scout).
