# Gen2 Corpus Backfill Plan

## Objective

Grow the Gen2 pinned-grid d02 24 h corpus from the current 3 complete members to at least 10 complete members so M7 production Tier-4 RMSE and tolerance work can run without surrogate data.

## Current Evidence

- M6-S7 reviewer reproduced the blocker: only 3 complete pinned-grid d02 members were usable, one additional complete run had the wrong grid, and the other 22 run directories lacked complete `wrfout_d02` history (`.agent/sprints/2026-05-21-m6-s7-tier4-probtest/reviewer-report.md:16`, `:25`, `:27`).
- M6.5-D1 manager closeout states that the loader and RMSE adapter are ready, but the corpus remains 3 complete 24 h d02 runs plus 22 partial runs, and more complete runs are needed for production Tier-4 (`.agent/sprints/2026-05-22-m6-5-d1-gen2-data-backfill/manager-closeout.md:54`, `:58`).
- The available scaffold expects a minimum sample size of 10 for the M6-S7/M7 Tier-4 path (`.agent/sprints/2026-05-21-m6-s7-tier4-probtest/reviewer-report.md:50`, `:107`).

## Target State

- Minimum: 10 complete pinned-grid d02 24 h members, all on the current `d02` mass shape `(66, 159)` / WRF extent `(67, 160)`.
- Preferred: 14 complete members to leave margin for one or two failed or non-pinned cycles before the M7-S0 Tier-4 harness freezes tolerances.
- Each retained run must include `wrfinput_d01`, `wrfinput_d02`, `wrfbdy_d01`, `namelist.input`, `namelist.output`, and hourly `wrfout_d02_*` from +0 through +24 h.
- The held-out cycle for validation must not be used to derive tolerances.

## Owner And Coordination

- Execution owner: Canairy Gen2 operator/team, because the CPU WRF production/backfill tree lives under `/mnt/data/canairy_meteo/**`, which this project treats as read-only.
- Project owner: M7 manager/human arbiter coordinates the retention request and confirms available disk.
- GPU project worker role: read-only inventory and RMSE harness only; no mutation under `/mnt/data/canairy_meteo/**`.
- Coordination channel: manager/human instruction to the local Canairy Gen2 run owner for `/home/enric/src/canairy_meteo/Gen2/` and `/mnt/data/canairy_meteo/runs/`.

## Backfill Path

1. Retain complete d02 history for the next 7 or more daily 18Z Gen2 runs.
   - Change the Gen2 retention policy so `wrfout_d02_*` is not stripped after the run completes.
   - First target window: the next 7 successful 18Z cycles after this plan.
   - Stop condition: at least 10 complete pinned-grid d02 members pass inventory.

2. If the next 7 daily live runs are not enough, rerun recent missing cycles.
   - Use the existing Gen2 WPS cases under `/mnt/data/canairy_meteo/runs/wps_cases/<cycle>_18z_72h/` where AIFS/WPS artifacts already exist.
   - Prefer cycles with current pinned grid and complete d01/d02 met_em coverage.
   - Retain only the required 24 h d02 history plus normal run metadata if disk is tight.

3. Re-run the existing inventory and quality checks from M6.5-D1.
   - Use the existing Gen2 loader and quality audit path.
   - Do not alter M6.5-D1 code in this M7-S0a sprint; that file ownership is separate.
   - Confirm that all selected runs are pinned-grid and complete before any tolerance freeze.

4. Dispatch M7-S0 Tier-4 RMSE harness only after M6.x green evidence exists.
   - The harness can consume `compute_rmse_against_gen2` immediately, but any model-validity claim remains BLOCKED until M6.x lands.

## Wall Estimate

- Best case: 7 to 8 days elapsed, if Gen2 daily 18Z runs continue succeeding and retention is changed immediately.
- Conservative case: 10 to 14 days elapsed, allowing for late AIFS, failed CPU WRF runs, or a disk cleanup pass.
- Manual backfill case: 1 to 3 operator days, plus CPU WRF wall time, if the team chooses to rerun historical cycles instead of waiting for live accumulation.

## Blockers

- Disk space under `/mnt/data/canairy_meteo/runs/` for complete hourly d02 files.
- Gen2 production policy may intentionally strip d02 history; that policy must change or be bypassed for selected M7 backfill cycles.
- AIFS late/missing cycles can reduce daily yield.
- M6.x dycore evidence is still required before GPU-vs-Gen2 RMSE can be interpreted as model evidence.
- One previously complete member used an old `(66, 120)` d02 grid and must remain excluded.

## Proof Objects To Produce In Follow-up

- `gen2_backfill_inventory.json`: selected run IDs, paths, grid shape, file counts, mtimes, and exclusion reasons.
- `gen2_quality_audit.json`: M6.5-D1 quality checks over the selected members.
- `tier4_member_manifest.json`: no-peek training and held-out split consumed by M7-S0.
- `backfill_retention_note.md`: manager/human confirmation of retention policy and disk owner.

## Decision Needed

The manager/human must choose between waiting for 7+ future daily Gen2 cycles with retained d02 output or asking Canairy Gen2 to rerun recent pinned-grid cycles now. Waiting is lower risk operationally; rerun is faster if CPU WRF capacity and disk are available.
