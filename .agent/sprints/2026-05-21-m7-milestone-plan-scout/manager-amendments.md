# M7 Milestone Plan — Manager Amendments (Integrating Codex Critic)

**Manager**: Claude Opus 4.7 (1M-context)
**Date**: 2026-05-21 14:00
**Inputs**:
- M7 scout plan: `.agent/sprints/2026-05-21-m7-milestone-plan-scout/m7-milestone-plan.md` (34 KB, 9 sprints S0-S8)
- M7 codex critic: `.agent/sprints/2026-05-21-m7-milestone-plan-scout/critical-review-codex.md` (RATIFY-WITH-AMENDMENTS, 12 edits + 10 risk-register additions)

## Verdict: RATIFY-WITH-AMENDMENTS (final)

Adopt all 12 critic amendments + integrate 10 risk-register additions. Sequencing + proof schemas + claim scope are now load-bearing.

## Twelve binding amendments (per critic §10)

1. **S0 ADR-011 reality check**: scout assumed ADR-011 absent; in fact M6-S2a closed and ADR-011 exists. Update S0 to gate on REVIEWED M6 closeout status, M5 radiation closeout (still in flight via M5-S3.zzzz + S3.zzz), and variable-specific GREEN/PARTIAL/BLOCKED/FAIL status per M6-S2/S3/S5/S8 proof objects.
2. **S0 read-only if overlapping with M6**: do NOT append to `src/gpuwrf/io/proof_schemas.py` until M6 schemas frozen.
3. **Freeze shared M7 interfaces in S0/S1**: run manifest, cycle lifecycle, status state machine, output/product manifest, proof-schema registry, station-collocation API, atomic publish contract. Assign ONE owning sprint per.
4. **S1 amendments**: WPS/AIFS command provenance, geog/static checksums, source license/version metadata, partial-GRIB failure schema, `GEN2_SAME_CYCLE_MISSING` status. May split ingest/preflight from full 3km run (36-60h is borderline).
5. **Sequencing amendment**: S4/S5/S6/S7 can SCAFFOLD in parallel but FINAL ACCEPTANCE is serial through S1 24h output → S4 product manifest → S5 scores → S6 recovery → S7 live examples → S8 closeout.
6. **S2 1km PASS gate rewrite**: scout focused on raw HBM (32 GB); actual risk is XLA compile/temporary/retrace/cache. Add HLO/StableHLO size, op count, compile retries, retrace count, temp peak, cache size, allocator fragmentation.
7. **S3 1km claim scope**: d03-only = `PARTIAL_TENERIFE_ONLY`; full "Canary 1 km" requires d03+d04+d05 OR explicit manager-approved deviation.
8. **S5 live-observation requirements**: freshness windows, API credentials (AEMET/GRAFCAN), licensing, rate limits, wind speed/direction → U/V conversion, station height/elevation masks. Existing station cube ends 2026-05-07 — stale for live cycles post-21st. Add `BLOCKED_OBS_SOURCE` status.
9. **Validation gate operational**: U10/V10/T2 binding gate `gpu_vs_gen2_rmse ≤ gen2_vs_obs_rmse` (per validation philosophy). Q2/RH2 and precip stay PARTIAL/diagnostic unless coverage + event sample proven.
10. **S6 restart tests expanded**: process death, stale lock cleanup, compile-cache reload/corruption fallback, version/hash compatibility, crash injection at ingest/forecast/postprocess/verify/publish. WRF-compatible `wrfrst` write may close as explicit DEVIATION if non-blocking.
11. **Workstation operations gates**: disk quota + cleanup, GPU health preflight, single-machine unavailable state, UTC-only cycle IDs, cache writeability. NO 08:00 SLA claim until S8 soak proves it.
12. **S8 soak clarification**: explicitly whether three pinned cycles may be REPLAY or require three LIVE daily cycles. If live, calendar minimum is 3 days, not 12-24h + unspecified wait.

## Risk register additions (per critic §9)

Encoded for inclusion in M7-S0 contract risk section:

| Risk | Mitigation |
|---|---|
| Disk full under `/mnt/data/wrf_gpu2/operational/` | S0 retention quota; S7 disk_low threshold; S6 no-space crash test |
| Single-machine SPOF | S7 health preflight + stale-last-good + `MACHINE_UNAVAILABLE` status |
| Long-running JAX/driver instability | S8 soak per-cycle memory/compile/wall metrics; S6 process-kill recovery |
| AIFS upstream license/availability change | S0/S1 license/version manifest; stale-last-good fallback only |
| Live station API credentials/rate limits | S5 credential manifest + freshness gate + `BLOCKED_OBS_SOURCE` |
| Same-cycle Gen2 CPU run unavailable | Separate `GPU_CYCLE_OK_GEN2_MISSING` from forecast failure |
| WPS/Gen2 tooling drift | S1 pin command logs + executable paths + env vars + checksums |
| UTC/local-time cycle confusion | UTC-only cycle IDs everywhere |
| Nested 1km partial-claim ambiguity | `one_km_claim_scope` field + d03+d04+d05 for full Canary L3 |
| Output atomics across filesystems/symlinks | S1/S6 atomic publish contract + crash injection |

## Updated M7 sprint sequence (post-amendments)

| Sprint | Wall | Critical-path |
|---|---:|---|
| **M7-S0 prologue** (with all critic amendments to S0 baked in) | 12-18h | serial gate; M6 closeout review + radiation-closeout status + interface freeze |
| **M7-S1a AIFS ingest + WPS preflight** (split from S1 per critic) | 12-20h | serial after S0 |
| **M7-S1b 3 km daily forecast driver** | 24-36h | serial after S1a |
| M7-S2 1 km memory + XLA compile audit | 12-20h | parallel after S1b |
| M7-S3 1 km pipeline OR deviation | 36-72h pass / 12-18h fail | conditional after S2 |
| M7-S4 post-processing | 18-30h | scaffold parallel; final after S1b |
| M7-S5 live verification | 24-36h | scaffold parallel; final after S1b 24h output |
| M7-S6 restart + crash recovery (expanded) | 18-30h | scaffold parallel; final after S1b 12h smoke |
| M7-S7 monitoring + alerting | 12-24h | scaffold parallel after S0 |
| M7-S8 soak + closeout (live or replay) | 24-36h + soak wall (3 days if live) | serial close |

**Critical-path calendar updated**: 3km-only = 5-8 working days + 3-day live soak = ~8-11 days; with 1km = 8-12 working days + soak = ~11-15 days.

## Pre-dispatch decisions for M7-S0

Before M7-S0 dispatches:
1. Manager confirms M6 closeout status (waits for M6-S2 + M6-S3 + M6-S8 → GREEN)
2. M5 RRTMG PARITY status (M5-S3.zzzz + S3.zzz close)
3. Station observation source decision (live or replay-only)
4. 1km claim scope (Tenerife-only vs full Canary)

— Manager (Claude Opus 4.7 1M-context), 2026-05-21 14:00
