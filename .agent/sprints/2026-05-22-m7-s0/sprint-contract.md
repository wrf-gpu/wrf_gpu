# Sprint Contract - M7-S0 Tier-4 RMSE Harness

**Sprint ID**: `2026-05-22-m7-s0`
**Created**: 2026-05-22 by M7-S0a prologue
**Status**: **DRAFT - BLOCKED-on-M6.x**
**Dispatch condition**: M6.x dycore accepted GREEN or a reviewed c1 replacement accepted as the M7 model path. If M6 evidence is absent, this sprint may run schema/inventory preflight only and must emit `BLOCKED`.

## Objective

Build the first M7 Tier-4 RMSE harness that consumes M6.5-D1's frozen `compute_rmse_against_gen2` adapter, uses the Gen2 d02 corpus inventory without surrogate grids, and emits honest PASS/BLOCKED artifacts for GPU-vs-Gen2 model consistency. This is not station-observation operational verification; M7-S5 owns that claim.

## Acceptance

- **AC1 M6 inheritance gate**: produce `artifacts/m7/prologue/m6_inheritance_gate.json` with M6.x status, dycore path, M6-S2/S3 prerequisites, ADR-011 presence, proof-schema registry status, and blocking items. If M6.x is missing or failed, status is `BLOCKED`.
- **AC2 Gen2 corpus inventory**: produce `artifacts/m7/prologue/gen2_baseline_inventory.json` listing complete pinned-grid d02 24 h members, excluded partial/wrong-grid members, file counts, run paths, mtimes, and shape. Minimum PASS inventory is 10 complete pinned-grid members; fewer emits `BLOCKED_CORPUS`.
- **AC3 RMSE adapter call path**: call `gpuwrf.validation.data_quality.compute_rmse_against_gen2(gpu_forecast_state, gen2_wrfout_path, valid_time, fields=("U10", "V10", "T2"))` without reimplementing RMSE.
- **AC4 No-peek split**: freeze training/tolerance member list before held-out validation. The held-out cycle cannot update thresholds or selected members.
- **AC5 Tier-4 RMSE artifact**: produce `artifacts/m7/prologue/tier4_rmse_harness.json` with status, fields, leads, run IDs, RMSE records, adapter version/source, M6.x evidence pointer, and blocker list.
- **AC6 AIFS/WPS and station inputs consumed as readiness context**: read `data/manifests/aifs_ingest_v0.json` and `data/manifests/station_obs_sources_v0.json`; do not implement live AIFS or live station ingest in this sprint.
- **AC7 Tests**: add focused tests for inventory selection, blocked status when fewer than 10 complete members exist, adapter invocation shape, missing field error, and no-peek split.
- **AC8 Report**: worker report states whether the sprint is PASS or BLOCKED and separates model-consistency evidence from operational station verification.

## Files Worker May Modify

- `src/gpuwrf/validation/tier4_rmse_harness.py` (NEW)
- `scripts/m7_run_tier4_rmse_harness.py` (NEW)
- `tests/test_m7_tier4_rmse_harness.py` (NEW)
- `artifacts/m7/prologue/**` (NEW)
- `.agent/sprints/2026-05-22-m7-s0/**`

## Files Worker Must Not Modify

- `src/gpuwrf/dynamics/**`
- `src/gpuwrf/coupling/driver.py`
- `src/gpuwrf/contracts/state.py`
- `src/gpuwrf/physics/**`
- `/mnt/data/canairy_meteo/**`
- `src/gpuwrf/io/gen2_wrfout_loader.py`
- `src/gpuwrf/validation/data_quality.py`

## Dependencies

- M6.x dycore GREEN or reviewed replacement path.
- M6.5-D1 RMSE adapter accepted.
- Gen2 corpus complete enough for the requested gate, or the sprint exits `BLOCKED_CORPUS`.
- M7-S0a manifests and schemas present.

## Proof Objects

- `artifacts/m7/prologue/m6_inheritance_gate.json`
- `artifacts/m7/prologue/gen2_baseline_inventory.json`
- `artifacts/m7/prologue/tier4_member_split.json`
- `artifacts/m7/prologue/tier4_rmse_harness.json`
- `.agent/sprints/2026-05-22-m7-s0/worker-report.md`

## Hard Rules

1. No dynamics changes.
2. No writes under `/mnt/data/canairy_meteo/**`.
3. No RMSE reimplementation when `compute_rmse_against_gen2` covers the need.
4. No operational validation language; station-observation verification belongs to M7-S5.
5. If M6.x evidence or 10-member corpus is missing, emit a BLOCKED artifact rather than weakening acceptance.

## Dispatch

- Worker: codex gpt-5.5 xhigh
- Reviewer: Claude Opus 4.7 xhigh (mandatory)
- Wall-time: 12-18h if corpus exists; 2-4h to a clean BLOCKED artifact if M6.x or corpus is missing
- Branch: `worker/codex/m7-s0-tier4-rmse-harness`
