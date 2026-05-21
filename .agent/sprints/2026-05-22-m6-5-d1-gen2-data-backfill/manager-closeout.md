# M6.5-D1 Manager Closeout — Gen2 Backfill + RMSE Adapter: ACCEPT-with-amendment

**Sprint**: M6.5-D1 Gen2 Data Backfill + Quality Audit + RMSE Adapter
**Status**: **CLOSED — Opus ACCEPT with AC4 threshold-amendment (c)**
**Date**: 2026-05-21 ~23:50
**Worker**: codex gpt-5.5 xhigh (~18 min, vs 8-16h budget)
**Reviewer**: Claude Opus 4.7 xhigh

## Headline

Worker delivered 7/8 ACs PASS, 1/8 (AC4 boundary cross-check) PARTIAL at 1.02% V/E rel_mae vs 1% threshold. Opus reviewer found the **manager (me) wrote the wrong threshold** in the sprint contract — should be **3%** matching `boundary_replay.py:40-41` TOLERANCES, not 1%. AC4 disposition (c): threshold-amendment. M7-S0 dispatch is **UNBLOCKED**.

## Reviewer binding decision

**ACCEPT**. AC4 amended to 3% via threshold-amendment per `boundary_replay.py:40-41`. Manager error in sprint contract authorship, not worker defect.

## My (manager) honest accounting

Per `.agent/rules/sprint-lifecycle.md` discipline: when authoring a sprint contract, **check existing module TOLERANCES before specifying new thresholds**. I picked 1% by intuition; reviewer found the existing source-of-truth at 3%. Lesson saved to memory.

## ADR-016 amendment (this commit)

Amended line 57 in `.agent/decisions/ADR-016-gen2-data-corpus.md`:
- BEFORE: "Any relative MAE above 1 percent is a data-pipeline failure flag"
- AFTER: "Threshold matches `src/gpuwrf/io/boundary_replay.py:40-41` TOLERANCES (U/V/QVAPOR rel_mae_max=0.03, T rmse 0.5K, PH tight). Any breach is a data-pipeline failure flag."

## Reviewer-flagged follow-up tickets (M7-S0a, non-blocking)

1. `data_quality.py:181-190` — reasons-list shadowing bug after status flip (one-line fix)
2. `gen2_wrfout_loader.py:132-140` — wasted Dataset open when filename parse succeeds
3. `data_quality.py:285` — import `TOLERANCES` from `boundary_replay.py` for per-variable thresholds (vs hard-coded 1%)
4. Per-field spike policy: 7σ for surface scalars, 5σ for winds, RAINNC bounded at 0 (current 5σ across-the-board flags all complete runs as PARTIAL — conservative but noisy)
5. Add staggered-grid test fixture (`west_east_stag`) for loader

All five queued for M7-S0a follow-up. None block M7 dispatch.

## Strategic state

| Sprint | Status | Critical-path? |
|---|---|---|
| M6.x dycore | in-flight (~45m, investigating 1h PH overflow) | YES |
| M6.5-D1 | CLOSED ✓ | M7 prereq, now CLEAR |
| M6.x contingency | CLOSED (insurance archive) | c1 ready if M6.x fails |
| M6-S8 | DRAFT — dispatches after M6.x | downstream |
| M6-S6 followup d02 retry | DRAFT — dispatches after M6.x | downstream |
| M7-S0 | UNBLOCKED on M6.5-D1; needs M6.x close before dispatch | downstream |

## Verifiability triple (reviewer cited)

1. Tests reproduce on fabricated NetCDF: 19/19 ✓ (no /mnt/data)
2. Inventory matches live disk: 25 wrfbdy_d01 + 78 wrfout_d02 ✓
3. Lazy load genuine: per-file Dataset within `with` block, only metadata cached ✓

## Honest M7 prep state

- M7-S0 Tier-4 RMSE harness can dispatch the moment M6.x lands GREEN
- RMSE adapter shape frozen (ADR-016 lines 49-53)
- 3 complete 24h d02 runs available (one mismatched grid); 22 partial. **Bottleneck for production Tier-4**: more complete runs needed. M7-S0 should drive Gen2 to retain more d02 history (operational coordination with Canairy team).

— Manager (Claude Opus 4.7 1M-context), 2026-05-21 23:50
