# M7-S0a Manager Closeout — Operational/Data Readiness Prologue: ACCEPT-WITH-MINOR

**Sprint**: M7-S0a Operational/Data Readiness Prologue
**Status**: **CLOSED — Opus ACCEPT-WITH-MINOR**
**Date**: 2026-05-22 ~00:25
**Worker**: codex gpt-5.5 xhigh (~14 min, vs 12-18h budget)
**Reviewer**: Claude Opus 4.7 xhigh

## Headline

8/8 ACs PASS on substance. Reviewer ran 3-leg verifiability triple independently (tests reproduce, Gen2 paths verified via `ls`, station obs curl-checked). 3 non-blocking minor follow-ups identified. **M7-S0 dispatch UNBLOCKED on M6.x close + Gen2 corpus retention.**

## Deliverables landed

| AC | Item | Status |
|---|---|---|
| AC1 | AIFS/WPS ingest manifest (`data/manifests/aifs_ingest_v0.json`) | PASS |
| AC2 | Station obs source manifest (METAR AVAILABLE, AEMET/GRAFCAN PARTIAL, Gen2 cube LOCAL) | PASS |
| AC3 | OperationalOutput/Status/Scheduler + AIFSIngestManifest + StationObs schemas | PASS, 8/8 tests |
| AC4 | Gen2 corpus backfill plan (3 → 10+ complete runs) | PASS-WITH-MINOR (owner organizational) |
| AC5 | M7-S0 draft contract (BLOCKED-on-M6.x; consumes `compute_rmse_against_gen2`) | PASS ⭐ load-bearing |
| AC6 | M6-S8 renamed "operational" → "model-consistency" in-place | PASS ⭐ load-bearing |
| AC7 | 1km nest risk audit (real cells×bytes math) | PASS |
| AC8 | Critic findings tracker (23 PC-* findings extracted) | PASS-WITH-MINOR |

## 3 non-blocking follow-ups (queued)

- **F-S0a-1** (hygiene): add `!data/manifests/` exception to `.gitignore` so future contributors don't need `-f`
- **F-S0a-2** (cosmetic): directory rename `m6-s8-operational-closeout/` → `m6-s8-model-consistency-closeout/` if/when M6-S8 dispatches (churn-vs-clarity trade)
- **F-S0a-3** (clarity): tracker PC-12 disposition could note that ADR-016 code default (0.01 → TOLERANCES) follow-up is still open

Manager will apply F-S0a-1 immediately (1-line .gitignore edit); F-S0a-2 deferred to M6-S8 dispatch time; F-S0a-3 deferred to M7-S0a-2 tracker maintenance.

## Strategic state shift

This sprint validates the plan critic's PC-1 finding: **ops/data readiness in parallel with dycore work saves 12-18h of serialization waste**. Adopted cadence going forward.

## M7-S0 dispatch impact

**UNBLOCKED on M6.x close + Gen2 corpus retention decision**. The remaining gates are dycore-side (M6.x or c1) and human-side (Canairy team coordination for Gen2 retention). Neither is M7-S0a's responsibility.

## Verifiability triple (reviewer ran independently)

1. `pytest -q tests/test_m7_s0a_schemas.py` → 8 passed ✓
2. All cited Gen2 paths exist (9 paths `ls`-verified) ✓
3. METAR live curl HTTP 200 with sample obs; AEMET/GRAFCAN PARTIAL labels honest ✓

## What carries into M7-S0

- `compute_rmse_against_gen2` adapter signature pinned (M7-S0 AC3 will consume verbatim)
- Manifest schemas live: AIFSIngestManifest, StationObservationSourceManifest, OperationalOutput, OperationalStatus, OperationalScheduler
- BLOCKED-emission paths defined for both M6.x missing and BLOCKED_CORPUS (<10 members)
- 1km nest risk audit captures memory/compile risk ahead of M7-S2 dispatch
- Critic findings tracker is the running disposition record for all 23 PC findings

— Manager (Claude Opus 4.7 1M-context), 2026-05-22 00:25
