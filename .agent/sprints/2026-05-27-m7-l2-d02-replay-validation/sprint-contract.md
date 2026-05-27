# Sprint Contract — M7 L2 d02 Replay Validation (3km Island Nest)

**Sprint ID**: `2026-05-27-m7-l2-d02-replay-validation`
**Created**: 2026-05-27 (user direction: nested-grid validation before backfills)
**Status**: READY
**Predecessor**: `.agent/decisions/MILESTONE-M7-CLOSEOUT.md` (PIPELINE_GREEN on L3 d02 3km)

## Objective

The Gen2 L2 (`runs/wrf_l2/*_18z_l2_72h_*`) configurations are 2-domain nested runs:
- **L2 d01**: 9km, 94×60 grid (full Canary + ocean parent)
- **L2 d02**: 3km, 160×67 grid (island nest)

L2's d02 grid (160×67) **matches** the L3 d02 grid we already validated (the PIPELINE_GREEN one). Physical grid is the same; the difference is the boundary forcing source: L3 d02 uses Gen2 d01 outputs as boundary (single-domain Canary 3km driver), L2 d02 uses L2's own d01 (9km parent) outputs as boundary (one-way nest).

This sprint validates that our GPU port produces accurate 3km d02 forecasts when fed L2-d01-derived boundary conditions, demonstrating one-way nest compatibility on the operational data flow. **No new dycore code needed** — just point `build_replay_case` at L2 directories instead of L3.

## Acceptance

- **AC1 — L2 inventory**: enumerate complete L2 runs (both d01 and d02 wrfout present). Emit `.agent/sprints/2026-05-27-m7-l2-d02-replay-validation/l2_inventory.json` with: run_id, d01 file count, d02 file count, expected hours, grid shapes, time coverage.

- **AC2 — Replay case adapter for L2**: extend `gpuwrf.integration.d02_replay.build_replay_case` (or add a thin wrapper `build_l2_d02_replay_case`) to accept an L2 run directory. The function must:
  - Load L2 d02's `wrfinput_d02` as IC (or substitute with a wrfout snapshot at t=0)
  - Load L2 d01's hourly wrfouts as boundary tendency source for d02
  - Return the same `(state, namelist, grid, ...)` shape the existing L3 path produces
  - Verify grid shape matches L3 d02 expected (160×67 → mass shape 66×159; should be drop-in)

- **AC3 — GPU 24h forecast on L2 d02**: pick one L2 run with complete data (suggest `20260521_18z_l2_72h_*` or the most recent complete L2 day). Run `scripts/m7_daily_pipeline.py` against the L2 replay case for 24h. Emit hourly wrfouts at `/tmp/m7_pipeline_runs/l2_d02_<run_id>/`.

- **AC4 — Tier-4 RMSE vs L2 d02 reference**: compare GPU output to L2's own d02 wrfouts on T2/U10/V10 using `gpuwrf.validation.data_quality.compute_rmse_against_gen2`. Bounds: T2 ≤ 3K, U10/V10 ≤ 7.5 m/s (same envelope as M6 close). Emit `.agent/sprints/2026-05-27-m7-l2-d02-replay-validation/tier4_rmse_l2_d02.json`.

- **AC5 — Bounds + finiteness**: all output fields finite, per-level theta and wind bounds preserved per `feedback_validation_philosophy.md`. Emit `.agent/sprints/2026-05-27-m7-l2-d02-replay-validation/bounds_check_l2_d02.json`.

- **AC6 — Wall-clock measurement**: same per-step + 24h wall-clock measurement as the L3 d02 pipeline produced. Emit `.agent/sprints/2026-05-27-m7-l2-d02-replay-validation/wall_clock_l2_d02.json`.

- **AC7 — Worker report**: verdict `L2_D02_GREEN` / `L2_D02_BOUNDED_FAIL` / `L2_D02_BLOCKED` with the publish-readiness conclusion for nested-grid backfills.

## Files Worker May Modify

- `src/gpuwrf/integration/d02_replay.py` (extend with `build_l2_d02_replay_case` or generalize parent-dir argument; NO behavior change to existing L3 path)
- `scripts/m7_l2_d02_replay.py` (NEW — orchestrator)
- `tests/test_m7_l2_d02_replay.py` (NEW)
- `.agent/sprints/2026-05-27-m7-l2-d02-replay-validation/**`

## Files Worker Must Not Modify

- `src/gpuwrf/runtime/operational_mode.py` — frozen post-M7
- `src/gpuwrf/dynamics/**`, `src/gpuwrf/physics/**`, `src/gpuwrf/coupling/**` — frozen
- `src/gpuwrf/io/wrfout_writer.py` — frozen
- `src/gpuwrf/validation/forecast_vs_obs.py`, `data_quality.py`, `tier4_rmse_harness.py` — frozen
- governance files
- `/mnt/data/canairy_meteo/**` — read-only

## Hard Rules

1. **No dycore code changes.** L2 d02 grid is identical to L3 d02; this is purely a data-loading + driver-config sprint.
2. **GPU usage**: yes — runs 24h forecasts. Will share GPU with the honest-speedup sprint, which is mostly CPU-side analysis after AC3 of THIS sprint kicks off. Should be fine sequentially.
3. **CPU pinning**: `taskset -c 0-3` for the orchestrator.
4. **No remote push.** Local commit on `worker/gpt/m7-l2-d02-replay-validation` only.
5. **Honest BLOCKED**: if L2's d01 wrfouts can't be ingested as boundary source (grid mismatch, missing fields), emit BLOCKED with diagnosis instead of fudging.
6. **Do not interfere with tmux `0:1`** (nightly WRF on cores 4-31).
7. **Output dir**: large wrfouts to `/tmp/m7_pipeline_runs/l2_d02_*/`, not committed; JSON proofs to `.agent/sprints/.../`.

## Dependencies

- M7 perf-measurement step closed
- L2 backfill at `/mnt/data/canairy_meteo/runs/wrf_l2/` (verified: 27 day directories)
- L2 d01 wrfouts present for the chosen day (sprint verifies in AC1)
- GPU available (RTX 5090)

## Proof Objects

- `.agent/sprints/2026-05-27-m7-l2-d02-replay-validation/l2_inventory.json` (AC1)
- `.agent/sprints/2026-05-27-m7-l2-d02-replay-validation/tier4_rmse_l2_d02.json` (AC4 — gate)
- `.agent/sprints/2026-05-27-m7-l2-d02-replay-validation/bounds_check_l2_d02.json` (AC5)
- `.agent/sprints/2026-05-27-m7-l2-d02-replay-validation/wall_clock_l2_d02.json` (AC6)
- `.agent/sprints/2026-05-27-m7-l2-d02-replay-validation/worker-report.md` (AC7)
- `tests/test_m7_l2_d02_replay.py`

## Dispatch

- Worker: codex gpt-5.5 xhigh
- Wall-time: 3-5 h
- Branch: `worker/gpt/m7-l2-d02-replay-validation`
- Worktree: `/tmp/wrf_gpu2_l2d02`
- GPU usage: YES (24h forecast + comparison)
