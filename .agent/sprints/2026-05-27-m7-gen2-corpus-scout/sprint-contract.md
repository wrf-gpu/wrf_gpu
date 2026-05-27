# Sprint Contract — M7 Gen2 Corpus Scout (research-only)

**Sprint ID**: `2026-05-27-m7-gen2-corpus-scout`
**Created**: 2026-05-27 (autonomous overnight loop, parallel to pipeline integration)
**Status**: READY — RESEARCH ONLY (no code)
**Predecessor**: `.agent/sprints/2026-05-22-m7-s0/` (BLOCKED_CORPUS — 2/10 complete pinned-grid Gen2 d02 24h members)

## Objective

The M7 Tier-4 RMSE harness (`scripts/m7_run_tier4_rmse_harness.py`) blocked at corpus availability: only 2 of 10 required pinned-grid Gen2 d02 24h members were found. M7 gates #1 (IC/BC mapping proof) and #7 (full Tier-4 ensemble) both depend on a complete corpus.

This sprint is a **research-only opus-tester scout** that catalogs all available Gen2 wrfouts, identifies why the corpus gate fails, and produces an actionable recommendation: (a) which existing runs can be relabeled / reformatted into the required schema; (b) which fresh CPU WRF runs (if any) are needed; (c) whether the corpus requirement itself should be revised downward; (d) whether the d02 grid pinning can be relaxed without weakening the Tier-4 RMSE claim.

No code is written. No CPU WRF runs are triggered. No modification of `/mnt/data/canairy_meteo/`. Pure investigation.

## Acceptance

- **AC1 — Full Gen2 wrfout inventory**: enumerate every `wrfout_d0[1-5]_*` file under `/mnt/data/canairy_meteo/runs/wrf_l[23]/` and `/mnt/data/canairy_meteo/runs/wrf_l3/`. For each: run-ID, domain (d01-d05), valid time, file size, grid shape (via `netCDF4` header read, not full load), is_complete (covers full 24h with hourly outputs). Emit `.agent/sprints/2026-05-27-m7-gen2-corpus-scout/full_gen2_inventory.json`.

- **AC2 — Pinned-grid analysis**: the M7-S0 harness requires "complete pinned-grid d02 24h members." From AC1, classify each d02 run into one of: PINNED_GRID_COMPLETE, PINNED_GRID_INCOMPLETE, WRONG_GRID, MISSING_TIMES. Identify the modal grid shape across d02 runs; identify which shape is "the pinned grid" expected by the M7-S0 harness. Cross-reference with `src/gpuwrf/validation/data_quality.py:compute_rmse_against_gen2` adapter to confirm the pinning logic. Emit `.agent/sprints/2026-05-27-m7-gen2-corpus-scout/pinning_analysis.md`.

- **AC3 — Recovery candidates**: for each non-PINNED_GRID_COMPLETE member, name the specific reason. Top candidates: (a) a particular hour's wrfout is missing; (b) the grid was rerun at a different size; (c) the run failed mid-forecast. For each, indicate whether the issue is recoverable by data-side action (no fresh forecast needed). Emit `.agent/sprints/2026-05-27-m7-gen2-corpus-scout/recovery_candidates.md`.

- **AC4 — Recommendation**: write `.agent/sprints/2026-05-27-m7-gen2-corpus-scout/recommendation.md` with concrete next-step options:
  - **Option A**: lower the M7-S0 corpus floor (e.g., 5 instead of 10) — what's the statistical cost?
  - **Option B**: relax the pinned-grid requirement — what's the comparator complexity cost?
  - **Option C**: relabel WRONG_GRID runs as a separate grid family — does this still meet the spirit of the M7 acceptance gate?
  - **Option D**: trigger fresh CPU WRF runs (out-of-scope for this sprint but recommend the day count + estimated wall-time)
  - **Recommended**: pick one with reasoning.

- **AC5 — Tester report** with verdict `RECOMMENDATION_READY` / `BLOCKED_NO_RECOVERY`. The verdict drives whether the manager dispatches a follow-up sprint (writer/relabeler) or escalates to the user.

## Files Tester May Read

- All Gen2 directories under `/mnt/data/canairy_meteo/runs/` (read-only; no writes ever)
- All AIFS files under `/mnt/data/canairy_meteo/data/aifs_single/`
- `src/gpuwrf/validation/data_quality.py`, `src/gpuwrf/validation/tier4_rmse_harness.py`
- `src/gpuwrf/integration/d02_replay.py`
- `artifacts/m7/prologue/*.json` (the M7-S0 outputs)
- M7 milestone files

## Files Tester May Modify

- `.agent/sprints/2026-05-27-m7-gen2-corpus-scout/**` only

## Hard Rules

1. **No code changes.** Pure scouting.
2. **No CPU WRF jobs triggered.** AC4 may name them as future work, not run them.
3. **No writes under `/mnt/data/canairy_meteo/`.**
4. **CPU pinning**: `taskset -c 0-3`.
5. **Do not interfere with tmux `0:1`** (nightly WRF).
6. **Do not load full wrfout files** — header reads only via `netCDF4.Dataset(path)`; do not iterate variable arrays. Goal is inventory speed, not data load.
7. **No memory updates** without manager approval.

## Dependencies

- iocompat audit complete (knows the reference grid shape)
- AIFS month files visible on disk (verify by listing `/mnt/data/canairy_meteo/data/aifs_single/`)

## Proof Objects

- `.agent/sprints/2026-05-27-m7-gen2-corpus-scout/full_gen2_inventory.json` (AC1)
- `.agent/sprints/2026-05-27-m7-gen2-corpus-scout/pinning_analysis.md` (AC2)
- `.agent/sprints/2026-05-27-m7-gen2-corpus-scout/recovery_candidates.md` (AC3)
- `.agent/sprints/2026-05-27-m7-gen2-corpus-scout/recommendation.md` (AC4)
- `.agent/sprints/2026-05-27-m7-gen2-corpus-scout/tester-report.md` (AC5)

## Dispatch

- Tester: claude opus 4.7 xhigh
- Wall-time: 1-3 h (research, single-day in scope)
- Branch: `tester/opus/m7-gen2-corpus-scout`
- Worktree: `/tmp/wrf_gpu2_corpus`
- GPU usage: NONE (header reads only)
