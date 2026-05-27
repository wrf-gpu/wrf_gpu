# Sprint Contract — M7 L2 Nest Scout + 9km d01 Architecture (research)

**Sprint ID**: `2026-05-27-m7-l2-nest-scout`
**Created**: 2026-05-27 (user direction: nested-grid capability assessment before backfills)
**Status**: READY — RESEARCH ONLY (no model code changes)
**Predecessor**: `.agent/decisions/MILESTONE-M7-CLOSEOUT.md`; L2 nest structure (d01 9km + d02 3km)

## Objective

The user wants to start operational backfills on L2 (9km parent + 3km island nest) today. The L2 d02 (3km) is validated by the parallel sprint `2026-05-27-m7-l2-d02-replay-validation` since L2 d02 grid matches L3 d02. The **open question** is the **9km d01 parent**: does our GPU port handle a 9km-resolution single-domain forecast, and what would a full backfill pipeline (run d01 then d02 one-way nested) look like?

This sprint is an opus research scout. No code changes. Produces an architecture decision and a backfill execution plan.

## Acceptance

- **AC1 — L2 day inventory**: list every L2 day directory at `/mnt/data/canairy_meteo/runs/wrf_l2/`. For each: completeness (both d01 + d02 wrfout series present), expected vs actual file counts, grid shapes (header-read via `netCDF4`), AIFS source presence. Emit `.agent/sprints/2026-05-27-m7-l2-nest-scout/l2_day_inventory.json`.

- **AC2 — 9km d01 feasibility audit**: read the existing dycore code (`src/gpuwrf/runtime/operational_mode.py`, `src/gpuwrf/dynamics/core/acoustic.py`, `src/gpuwrf/contracts/state.py`) and answer: is `dx` truly a configurable parameter, or are there hard-coded assumptions about 3km grid spacing anywhere? Search for `3000`, `dx_m`, grid-spacing assumptions in physics couplers. Document each finding (file:line + assessment). Emit `.agent/sprints/2026-05-27-m7-l2-nest-scout/9km_feasibility_audit.md`.

- **AC3 — L2 d01 boundary forcing source**: L2's d01 9km parent is driven by AIFS (not by a higher parent). Determine: how does Gen2's CPU WRF construct L2 d01 boundary tendencies? Inspect `wrfbdy_d01` schema, AIFS interpolation logic in `~/src/canairy_meteo/Gen2/` if accessible (read-only). Identify whether our existing `build_replay_case` can be extended to consume an AIFS-driven d01 wrfbdy, or whether a new IC/BC ingestor is needed. Emit `.agent/sprints/2026-05-27-m7-l2-nest-scout/d01_boundary_forcing_audit.md`.

- **AC4 — One-way nest backfill pipeline design**: write `.agent/sprints/2026-05-27-m7-l2-nest-scout/nest_backfill_design.md` describing the proposed daily backfill flow:
  - Step 1: ingest AIFS for day D → produce d01 wrfinput + wrfbdy
  - Step 2: run GPU on d01 9km for 24h or 72h → produce d01 wrfouts (1-hourly)
  - Step 3: use d01 wrfouts as boundary forcing for d02 (same path as L3 d02 replay)
  - Step 4: run GPU on d02 3km → produce d02 wrfouts
  - Step 5: write NetCDF outputs, score vs AEMET, archive
  Total wall-time estimate per day (extrapolated from L3 d02 ~5.4 min/24h; d01 is ~22% of d02's cells but at 9km so larger time-step possible → likely 2-4 min/24h).

- **AC5 — Risk assessment**: write `.agent/sprints/2026-05-27-m7-l2-nest-scout/risk_assessment.md` listing the **top 5 risks** for the user's "publish results, start backfills today" timeline. Each risk: severity, mitigation, owner.

- **AC6 — Tester report**: verdict `BACKFILL_PLAN_READY` / `BACKFILL_NEEDS_NEW_CODE` / `BACKFILL_BLOCKED` with the concrete next-sprint recommendation (if any code changes are needed) or the green-light recommendation (if backfills can start today).

## Files Tester May Read

- All of `src/gpuwrf/**`, `scripts/**`
- `/mnt/data/canairy_meteo/runs/wrf_l2/**`, `runs/wrf_l3/**`, `data/aifs_single/**` (READ-ONLY)
- `~/src/canairy_meteo/Gen2/**` if accessible (read-only Gen2 source to understand AIFS→d01 path)
- Governance + sprint contracts

## Files Tester May Modify

- `.agent/sprints/2026-05-27-m7-l2-nest-scout/**` only

## Hard Rules

1. **No code changes.** Pure scouting.
2. **No CPU WRF jobs triggered.** AC4 may name them as future work, not run them.
3. **No writes under `/mnt/data/canairy_meteo/`**.
4. **CPU pinning**: `taskset -c 0-3`.
5. **Do not interfere with tmux `0:1`** (nightly WRF).
6. **No memory updates** without manager approval.
7. **Honest BACKFILL_NEEDS_NEW_CODE**: if 9km requires real code work (e.g. AIFS→9km interpolation, new wrfbdy ingestor), say so plainly. The user accepts "fix it if needed" — but only based on a real diagnosis, not aspirational claim.

## Dependencies

- M7-CLOSEOUT complete
- L2 backfill on disk (27 days verified)
- netCDF4 + xarray Python stack

## Proof Objects

- `.agent/sprints/2026-05-27-m7-l2-nest-scout/l2_day_inventory.json` (AC1)
- `.agent/sprints/2026-05-27-m7-l2-nest-scout/9km_feasibility_audit.md` (AC2)
- `.agent/sprints/2026-05-27-m7-l2-nest-scout/d01_boundary_forcing_audit.md` (AC3)
- `.agent/sprints/2026-05-27-m7-l2-nest-scout/nest_backfill_design.md` (AC4 — main deliverable)
- `.agent/sprints/2026-05-27-m7-l2-nest-scout/risk_assessment.md` (AC5)
- `.agent/sprints/2026-05-27-m7-l2-nest-scout/tester-report.md` (AC6)

## Dispatch

- Tester: claude opus 4.7 xhigh
- Wall-time: 1-3 h (research)
- Branch: `tester/opus/m7-l2-nest-scout`
- Worktree: `/tmp/wrf_gpu2_nestscout`
- GPU usage: NONE
