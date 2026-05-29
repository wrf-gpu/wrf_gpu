# Morning Report — 2026-05-27

**Status: 🎉 M7-OPERATIONALLY-CLOSED**

Good morning! Overnight autonomous loop brought the project home.

## TL;DR

The GPU-native NWP system runs a 24-hour Canary 3 km forecast in **5.4 minutes** wall-clock, at **156.82× the speed** of the 28-rank CPU WRF baseline on the same workstation. Headline target was 4-8×. Cleared by 20-40×.

- ✅ **6/8 M7 acceptance gates fully closed**
- ⚠️ **2/8 partial** — externally blocked by Gen2 d02 corpus availability with documented recovery path (Option D operator action)
- 🎯 **13 sprints landed overnight** in M7
- 🪙 All proof objects on disk, all merges into `manager-2026-05-23`

## Headline numbers (24h Canary 3km, 20260521 V3 IC)

| Metric | Value |
|---|---|
| **GPU 24h pipeline wall (full end-to-end)** | **324.78 s (5.4 min)** |
| GPU forecast-only wall | 310.27 s |
| GPU 1h warm wall (3km) | 5.71 s |
| **Speedup vs 28-rank CPU WRF (24h)** | **156.82×** |
| Preliminary 1h warm speedup | ~1900× |
| D2H inside loop | **0 copies / 0 bytes** (ADR-027 holds) |
| Restart at hour-12 checkpoint | bitwise PASS (max delta 0.0) |
| Repeatability (two full runs) | bitwise PASS |
| Reproducibility CV (3 warm runs) | 0.42% |
| 1km full-domain peak VRAM | 7.28 GB / 32 GB (78% headroom) |
| AEMET station scoring rows | 1,747 (finite T2/U10/V10 BIAS/RMSE/MAE) |
| Hourly NetCDF wrfouts produced | 24/24 readable |

## M7 acceptance gates

| # | Gate | Status |
|---|---|---|
| 1 | IC/BC mapping proof | **PARTIAL** (one Canary day demonstrated; full proof needs corpus growth via Option D) |
| 2 | I/O compatibility matrix | ✅ **DONE** |
| 3 | Restart-continuity | ✅ **DONE** (bitwise) |
| 4 | End-to-end 3km daily pipeline | ✅ **DONE** (PIPELINE_GREEN) |
| 5 | Wall-clock evidence vs CPU | ✅ **DONE** (156.82×) |
| 6 | Forecast-vs-obs verification | ✅ **DONE** (scaffold + integrated in pipeline) |
| 7 | Full Tier-4 ensemble | **PARTIAL** (corpus-blocked; probationary bridge now available) |
| 8 | 1km readiness + memory audit | ✅ **DONE** (78% headroom) |

## Overnight sprint ledger

13 sprints landed and merged into `manager-2026-05-23`:

1. M7 GPU profile prep — wall-clock + Nsight + D2H audit (initially BLOCKED-D2H false alarm)
2. M7 D2H probe — opus angle (architecture/contract)
3. M7 D2H probe — codex angle (JIT/source-line) — proved the "blocker" was profiler-window misplacement
4. M7 profiler-window fix — recapture confirmed **D2H = 0 inside loop**
5. M6c-01 mu regression — diagnostic merged; found production guards are load-bearing on 20260509
6. M7 1km memory audit — **FITS_WITH_HEADROOM**
7. M7 wrfout I/O compatibility audit — gap exposed (writer was .npz, not NetCDF)
8. M7 restart-continuity test — **bitwise PASS**
9. M7 NetCDF wrfout writer — **WRITER_READY** (0 critical gaps on 41-var minimum)
10. M7 forecast-vs-obs scaffold — **SCAFFOLD_READY** (106 AEMET stations)
11. **M7 daily-pipeline integration** — **PIPELINE_GREEN, 156.82× speedup**
12. M7 Gen2 corpus scout — **RECOMMENDATION_READY** (Option D + bounded A)
13. M7 corpus bridge — bounded Option A bridge + DEFAULT_M6_GEN2_RUN_DIR rebind

## Key proof objects

- `.agent/decisions/MILESTONE-M7-CLOSEOUT.md` — the close memo (commit `1820548`)
- `.agent/decisions/M7-PERF-MEASUREMENT-CLOSEOUT.md` — perf-step closeout
- `.agent/sprints/2026-05-27-m7-daily-pipeline-integration/pipeline_run_20260521.json` — PIPELINE_GREEN
- `.agent/sprints/2026-05-27-m7-daily-pipeline-integration/speedup_vs_cpu_24h.json` — 156.82×
- `.agent/sprints/2026-05-27-m7-profiler-window-fix/d2h_audit_v2.json` — D2H=0 invariant
- `.agent/sprints/2026-05-27-m7-1km-memory-audit/step_feasibility.json` — 1km fits
- `.agent/sprints/2026-05-27-m7-restart-continuity/restart_continuity.json` — bitwise restart
- `.agent/sprints/2026-05-27-m7-gen2-corpus-scout/recommendation.md` — Option D + bounded A
- All 13 sprint worker/tester reports + structured JSON artifacts

## What you (the principal) might want to decide

1. **Authorize Option D**: flip Gen2 retention on `~/src/canairy_meteo/Gen2/` and replay 5-7 missing cycles using the existing WPS staging dirs (`runs/wps_cases/{20260428,20260429,20260521,20260522,20260523,20260524,20260525}_18z_72h/`). After 2-4 nights, corpus grows ≥ 10 → gates #1 + #7 close to FULL → M7-CLOSED-FULL.

2. **Accept M7-OPERATIONALLY-CLOSED as-is**: the core technical claim is proven; release planning (M8) can begin without waiting for corpus.

3. **Engage M8 (public/forkable release)**: docs hygiene, license review, packaging. Substantial scope; needs principal direction on naming/IP.

4. **Tidy M6c carryforward**: 6 caveats from M6 closeout; none are M7 blockers. The `_m6b_acoustic_tendencies` identity shim removal is the smallest (5-line cleanup, now unblocked since pipeline lock released).

5. **Optimize further** (optional — already 20-40× over target): S1/S2/S3 from the D2H opus probe (mass_to_face FP32 chain, cuSPARSE → Thomas scan, _enforce_operational_precision hoist) — each a small fusion sprint.

## Caveats and risks (honest)

- **Probationary Option A bridge** ships at N=5 floor with `--non-operational` tag; current corpus is N=3 → bridge runs emit `PASS_PROBATIONARY_PENDING` (needs +2 cycles). The bridge does NOT change operational default (still N=10).
- **Station scoring is a measurement, not a CPU-skill claim**. Comparing GPU vs CPU skill side-by-side is a downstream sprint.
- **24h forecast not yet probed for stability across all 30 backfill days**. Verified on 20260521 (the V3 IC pin). Multi-day stress is corpus-bounded.
- **NetCDF writer covers 41-var minimum subset**. The full 362-variable WRF schema is not replicated (deliberately — most aren't consumed by Gen2 post-processing).
- **D2H ADR-027 invariant verified on 3km only**. 1km feasibility verified by one warm step; 24h × 1km not yet measured (audit recommends as separate sprint).
- **`test_m7_s0a_schemas.py` still failing** on a missing external file `namelist.wps`; this is data-side, not code-side; pre-existing before tonight; flagged in M7 closeout.

## CPU baseline note

The Gen2 nightly WRF in tmux 0:1 has been running healthy throughout the night (cores 4-31, no GPU). Current sim time approaching `2026-05-26 06:14` of day 1 — normal progress. The 156.82× speedup number comes from comparing GPU 24h wall vs Gen2 24h baseline timing reconstructed from `namelist.output` per-step records, not from a fresh CPU run.

## Suggested first actions

If you want to celebrate first: 🥂🚀 The project's headline target is cleared by 20-40×. **The GPU is faster than the CPU baseline by orders of magnitude.** That's the win.

If you want to keep moving: pick from items 1-5 above.

If you want to escalate to M8 (public release): pause for naming/IP review first per `LICENSE_NOTES.md`.

— Manager (Claude Opus 4.7, 1M context, autonomous overnight loop)
