# Sprint Contract — M7 Forecast-vs-Observation Verification Scaffold

**Sprint ID**: `2026-05-27-m7-forecast-vs-obs-scaffold`
**Created**: 2026-05-27 (autonomous overnight loop, parallel to restart-continuity + NetCDF writer)
**Status**: READY
**Predecessor**: `.agent/decisions/M7-PERF-MEASUREMENT-CLOSEOUT.md` (3km wall-clock done); 1km audit FITS; iocompat audit done

## Objective

Per M7 acceptance gate #6 (`.agent/milestones/M7-canary-operational-v0.md`): produce **forecast-vs-observation verification using T2 / wind / precip BIAS+RMSE plus one neighbourhood or object-based precip score.**

This sprint builds the **verification scaffold** by validating it against the existing Gen2 CPU WRF forecasts (which are known to be operationally useful). When the GPU forecast is later wired into the daily pipeline, the same scaffold scores it. The scaffold is the binding measurement infrastructure — NOT a GPU forecast claim.

Inputs available:
- **Existing CPU WRF wrfouts**: `/mnt/data/canairy_meteo/runs/wrf_l3/*/wrfout_d02_*` (~34 daily 3km runs, ~30 with complete coverage)
- **AEMET station observations**: `/mnt/data/canairy_meteo/artifacts/datasets/aemet_stations/*.parquet` (daily-aggregated per station) + `/mnt/data/canairy_meteo/truth/station_observations/tenerife_500m/station_observations.parquet`
- **Station metadata**: presumed within the parquet schema (lat, lon, elev, station id)

## Acceptance

- **AC1 — Observation inventory**: enumerate AEMET station files, schema, available variables, spatial coverage over the Canary domain. Emit `.agent/sprints/2026-05-27-m7-forecast-vs-obs-scaffold/aemet_observation_inventory.json` with: station count, schema (columns + dtypes), variables present (T2/wind/precip/etc.), temporal coverage, spatial bbox.

- **AC2 — Forecast-to-station interpolator**: implement `gpuwrf.validation.forecast_vs_obs.interpolate_to_stations(wrfout_path, station_metadata, *, variables, valid_time) → DataFrame` that produces per-station forecast values for the requested variables at the requested valid time. Bilinear horizontal interpolation; nearest vertical level for surface variables (T2, U10, V10 are 2D); no soil interpolation.

- **AC3 — BIAS/RMSE/MAE computer**: implement `gpuwrf.validation.forecast_vs_obs.compute_station_scores(forecast_at_stations, observations_at_stations, *, variables) → ScoreReport` returning per-variable BIAS, RMSE, MAE, sample count. Per-station scores aggregated across the observation window with `time` join. Handle missing data gracefully.

- **AC4 — Precip neighbourhood score**: implement a **Fractions Skill Score (FSS)** at a single neighbourhood scale (e.g. 9×9 grid cells) for precipitation. This is the "one neighbourhood or object-based precip score" the M7 gate calls out. Document the threshold + neighbourhood choice. Uses the Gen2 `RAINNC + RAINC` total accumulation between two `wrfout_d02_*` snapshots; observations from AEMET daily precip. (If precip observations are missing or sparse, document a BLOCKED_PRECIP outcome and emit a finite-but-unreliable score with the caveat.)

- **AC5 — Scaffold validation**: run the scaffold against ONE Gen2 wrfout day (pick a recent successful 3km run, e.g. `wrf_l3/20260520_18z_l3_24h_*`). Emit `.agent/sprints/2026-05-27-m7-forecast-vs-obs-scaffold/cpu_baseline_scaffold_run.json` with the resulting score report. This is **NOT** a claim that the CPU is good or bad — it's a sanity check that the scaffold computes finite numbers on real data.

- **AC6 — Tests**: add `tests/test_m7_forecast_vs_obs.py` with:
  - Synthetic 2D field + synthetic station table → known-answer interpolator test
  - Score-computer test with synthetic forecast/obs pairs producing known BIAS/RMSE
  - FSS test with synthetic precip patterns
  - Edge cases: missing stations, all-NaN obs, no temporal overlap

- **AC7 — Documentation**: write `.agent/sprints/2026-05-27-m7-forecast-vs-obs-scaffold/usage.md` showing how to invoke the scaffold against a wrfout — concrete CLI/API example.

- **AC8 — Worker report** with verdict `SCAFFOLD_READY` / `BLOCKED_DATA` / `PARTIAL`.

## Files Worker May Modify

- `src/gpuwrf/validation/forecast_vs_obs.py` (NEW)
- `src/gpuwrf/validation/__init__.py` (export)
- `scripts/m7_forecast_vs_obs_smoke.py` (NEW — runs the scaffold against one day)
- `tests/test_m7_forecast_vs_obs.py` (NEW)
- `.agent/sprints/2026-05-27-m7-forecast-vs-obs-scaffold/**`

## Files Worker Must Not Modify

- Existing `src/gpuwrf/validation/data_quality.py` (Tier-4 RMSE adapter — already frozen per M6.5-D1)
- `src/gpuwrf/**` outside validation/
- governance files
- `/mnt/data/canairy_meteo/**` — read-only

## Hard Rules

1. **No GPU runtime in this sprint.** All work is host-side pandas/xarray. Parallel-safe with restart-continuity (which uses GPU).
2. **CPU pinning**: `taskset -c 0-3`.
3. **No model code changes outside validation/.**
4. **Do not interfere with tmux `0:1`** (nightly WRF).
5. **No remote push.** Local commit on `worker/gpt/m7-forecast-vs-obs-scaffold` only.
6. **Read-only on Gen2/AEMET data**: under no circumstances write to `/mnt/data/canairy_meteo/`.
7. **One representative day for AC5** — do not bulk-iterate.
8. **Honest BLOCKED_DATA**: if AEMET schema doesn't actually contain T2/wind/precip in a usable form, emit BLOCKED_DATA with the schema findings. Do not fabricate.

## Dependencies

- iocompat audit complete (knows what fields exist in Gen2 wrfouts)
- AEMET station parquet files present (verified above)
- Standard Python stack: `pandas`, `xarray`, `netCDF4`, `numpy`, `pyarrow`

## Proof Objects

- `.agent/sprints/2026-05-27-m7-forecast-vs-obs-scaffold/aemet_observation_inventory.json` (AC1)
- `.agent/sprints/2026-05-27-m7-forecast-vs-obs-scaffold/cpu_baseline_scaffold_run.json` (AC5 — gate)
- `.agent/sprints/2026-05-27-m7-forecast-vs-obs-scaffold/usage.md` (AC7)
- `.agent/sprints/2026-05-27-m7-forecast-vs-obs-scaffold/worker-report.md` (AC8)
- `tests/test_m7_forecast_vs_obs.py` (AC6)

## Dispatch

- Worker: codex gpt-5.5 xhigh
- Wall-time: 3-6 h
- Branch: `worker/gpt/m7-forecast-vs-obs-scaffold`
- Worktree: `/tmp/wrf_gpu2_fvobs`
- GPU usage: NONE (parallel-safe)
