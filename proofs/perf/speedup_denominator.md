# CPU-WRF Speedup Denominator — rigorous, provenance-backed

> **⚠ SUPERSEDED (2026-06-12, v0.14):** the speedup multipliers below were
> measured on 2026-05-30 against an earlier, faster, **incomplete** dycore.
> Completing the fully WRF-faithful dycore + physics (v0.13/v0.14) raised
> per-step compute to **parity** with 28-rank CPU-WRF; the v0.14 measured
> end-to-end speedup is **~1.05×** (see `proofs/perf/v014_perf_regression_triage.json`
> and `docs/PERFORMANCE.md`). This file is retained as a dated historical
> record — its multipliers are **NOT** current v0.14 claims.

**Author:** opus frontrunner (`worker/opus/speedup-denominator`)
**Date:** 2026-05-30
**Mode:** READ-ONLY analysis of existing WRF timing logs. No new WRF runs, no GPU, no forecast. `taskset -c 0-3`.
**Purpose:** Replace the hand-waved "~50–85×" v0.1.0 speedup claim with a real, falsifiable number and explicit caveats.

---

## TL;DR headline

> **Honest apples-to-apples speedup (one RTX 5090 vs 28-rank CPU-WRF, same d02 grid, same model time, fp64): ~5×, with a defensible band of ~5–8× and a strict dt-parity floor of ~3.2×.**
> **The single biggest caveat: this is d02-ONLY (standalone single domain). The GPU port does NOT yet run the d01 parent or the d03–d05 1 km children. The previous "~50–85×" figure came from comparing one GPU d02 against the *entire multi-domain CPU nest wall* — apples-to-oranges and not honest.**

---

## 1. The numerator (GPU) — what we are dividing by

Source: `proofs/perf/segscan_24h.json` and `proofs/perf/warmed_timing.json` (this repo; tracked from commit `177c734`).

- Grid: **159 × 66 × 44** (= WRF `e_we=160, e_sn=67` mass points — identical grid to the CPU d02 below).
- Domain: **d02 standalone** (single domain; lateral boundary fed from the corpus d01→d02 boundary, not co-integrated).
- Device: one **RTX 5090**, **fp64**, `force_fp64=true`, guards disabled, flux advection, `epssm=0.5`, radiation cadence 180 steps.
- **dt = 10 s**, 10 acoustic substeps. 24 h = 8640 steps.
- Warmed throughput (compile excluded):
  - `segscan_24h.json`: 24 h warmed wall = 368.35 s → **15.35 s/forecast-hour** (≈ 42.6 ms/step). Extrapolates linearly: 48 h → 15.35, 72 h → 15.35 s/fc-hour. (The running-verdict's "L2 48h=15.69 / 72h=15.47" numbers are this same family, ±IO noise.)
  - `warmed_timing.json`: `warmed_ms_per_forecast_hour = 16391.9` → **16.39 s/forecast-hour** (45.5 ms/step). Same dt=10s.
- Peak GPU memory ≈ 9–10 GB.

**Reconciliation of the "47 ms/step elsewhere":** `segscan_24h.json` reports 42.6 ms/step; `warmed_timing.json` reports 45.5 ms/step; the verdict's ~47 ms is the same number plus per-call/IO overhead. All three collapse to **15.3–16.4 s/forecast-hour at dt=10s**. I use the band **15.35 (fast) – 16.39 (slow) s/fc-hour** below.

---

## 2. The denominator (CPU-WRF d02) — provenance

### Primary source: clean 2-domain L2 runs (the right comparison)

These are **finished 72 h runs where d02 is nested with ONLY d01 and has NO inner 1 km children** — the closest existing analog to a standalone d02, so the per-step `Timing for main on domain 2` lines isolate d02's own integration cleanly.

| Field | Value |
|---|---|
| WRF version | **V4.7.1** (banner in `rsl.out.0000`) |
| Ranks | **28** (`nproc_x=7, nproc_y=4`; 28 `rsl.error.*` files) |
| Domains | **max_dom=2**: d01 9 km 94×60, **d02 3 km 160×67×45** |
| d01 dt | `time_step=18` s |
| **d02 dt** | 18 / `parent_time_step_ratio=3` = **6 s** |
| Physics | CONUS suite, Thompson mp(8), MYNN PBL(5), Noah-MP(4), RRTMG(4), radt=30 min |

**Run A** — `run_id 20260528_18z_l2_72h_20260529T002423Z`
`/mnt/data/canairy_meteo/runs/wrf_l2/20260528_18z_l2_72h_20260529T002423Z/rsl.error.0000`
- d02 timing lines sampled: **43200** (= 72 h × 3600 / 6 s; full run, warmup-5 dropped).
- d02 **median = 0.1431 s/step**, trimmed-mean(5–95%) = 0.1603 s/step, p90 = 0.2524, max 41.4 s (radiation/IO spikes).
- d02 per-forecast-hour (median × 600 steps/fc-hr) = **85.9 s/fc-hour** (clean compute).
- d02 total-compute / 72 = **160.7 s/fc-hour** (includes all radiation + IO spikes).

**Run B** — `run_id 20260527_18z_l2_72h_20260528T002306Z`
`/mnt/data/canairy_meteo/runs/wrf_l2/20260527_18z_l2_72h_20260528T002306Z/rsl.error.0000`
- d02 lines: 43200. **median = 0.1338 s/step**, trimmed-mean = 0.1368.
- d02 median × 600 = **80.3 s/fc-hour**; total-compute / 72 = **84.5 s/fc-hour** (this run had far less IO/radiation contention, so the two metrics nearly coincide).

**Two independent runs agree: CPU d02 clean-compute ≈ 80–86 s/forecast-hour; realistic (incl. radiation+IO) ≈ 85–161 s/forecast-hour.**
Adopted denominators: **clean = 83 s/fc-hour**, **realistic = 123 s/fc-hour** (midpoints).

Full-nest wall sanity: Run A's full d01+d02 72 h wall (file mtimes `namelist.input` 08:56:38 → `rsl.error.0000` 13:17:03) ≈ 4 h 20 m = **216 s/fc-hour for the whole 2-domain nest** including all overhead. Consistent: d02 alone is ~83–161 s/fc-hr, d01 adds the rest.

### Cross-check / reference: live 5-domain L3 run (running NOW, read-only)

`/mnt/data/canairy_meteo/runs/wrf_l3/20260529_18z_l3_24h_20260530T054804Z/` — the 28-rank `prterun` job currently on cores 4–31 (PIDs 3518022+). I read its logs only and did not disturb it.
- **max_dom=5**: d01 9 km, **d02 3 km 160×67** (same as L2 d02), d03/d04/d05 1 km. d02 dt = 6 s.
- d02 median = 1.0445 s/step → 627 s/fc-hour — **~7× slower than the L2 d02** because the workstation is heavily contended (this box is simultaneously running GPU/other jobs, and the 5-domain nest competes for the same 28 cores). Full-nest wall per d02 fc-hour swung **11 min to 114 min** across hours.
- **This run is NOT a usable denominator** for a clean per-d02 number: its d02 timing is contaminated by contention and by sharing cores with d01+d03+d04+d05. It is included only to show *why the contended-nest wall is not a fair denominator*, and to confirm the d02 grid/dt config matches.

---

## 3. The honest speedup factor

All comparisons are **per forecast-hour (same model time)**, NOT per-step — because dt differs (CPU 6 s vs GPU 10 s). Per-step comparison is meaningless and is reported only to expose the trap.

### A. Apples-to-apples — GPU d02 vs CPU d02 (both standalone single-domain)

| | CPU d02 s/fc-hour | GPU d02 s/fc-hour | Speedup |
|---|---|---|---|
| **Conservative (lower bound)** | 83 (clean compute) | 16.39 (slow) | **5.1×** |
| **Midpoint** | 83 (clean compute) | 15.35 (fast) | **5.4×** |
| **Optimistic** | 123 (incl. IO/radiation, real wall) | 15.35 (fast) | **8.0×** |

**→ Defensible headline band: ~5–8×, central estimate ~5×.**

### B. Strict dt-parity floor

The GPU earns part of its win by being stable at **dt=10 s** while CPU-WRF runs **dt=6 s** (more steps). This is a legitimate efficiency of the port, so framing A correctly credits it. But if the GPU were forced to dt=6 s for strict per-timestep parity, its cost rises ~1.67× to ~25.6 s/fc-hour:
- 83 / 25.6 = **3.2× (hard floor, dt-matched)**.

### C. The inflated "~50–85×" — where it came from, why it is wrong

| Framing | Number | Verdict |
|---|---|---|
| Per-step CPU/GPU (dt mismatch ignored) | 0.143 / 0.0426 = **3.4×** | meaningless (different dt) |
| Full **5-domain** L3 nest median wall / GPU-d02 | 2559 / 15.35 = **167×** | apples-to-oranges (5 domains vs 1) |
| Sum-of-all-domain compute / GPU-d02 | 8317 / 15.35 = **542×** | wildly overstated |
| Full **2-domain** L2 nest (d01+d02 compute) / GPU-d02 | 377 / 15.35 = **24×** | overstated (GPU has no d01) |

The "~50–85×" sits in the gap between the 24× (L2 d01+d02 nest) and the 167× (L3 full nest) framings. **None of these is honest** because the GPU runs only d02. Do **not** headline any of them.

---

## 4. Caveats (state prominently in the paper)

1. **d02-ONLY / standalone.** The GPU port runs a single 3 km domain with prescribed boundaries. The CPU operational product is a 2- or 5-domain nest. The GPU does **not** produce d01 (9 km) or d03–d05 (1 km). The 5× is "one GPU replaces one CPU d02," not "one GPU replaces the operational nest."
2. **fp64 on both sides.** The GPU number is fp64. A planned **fp32 downcast** (`proofs/perf/fp32_downcast_plan.md`) is expected to roughly halve GPU time and push the apples-to-apples speedup toward ~8–12×, but that is a projection, not measured here.
3. **Single GPU vs 28 CPU ranks.** Denominator is the full 28-core workstation (cores 4–31). The 5× is per-socket-vs-per-GPU on the *same* box; it is not a per-watt or per-dollar claim.
4. **CPU d02 timing isolation.** The L2 `Timing for main on domain 2` line measures d02's own solver step; it excludes d01's separate timing but the d02 step is still driven by d01's lateral boundary each parent step (so is the GPU's). This is fair. The L3 (5-domain) d02 timing is **contaminated** by contention and core-sharing with d03–d05 and is not used as the denominator.
5. **dt asymmetry.** CPU 6 s vs GPU 10 s. Comparison is per-forecast-hour. The dt-matched floor is 3.2× (§3B).
6. **Workstation contention.** The L2 numbers are from finished overnight runs; the live L3 run shows the same box can be 7× slower under contention. The 83 s/fc-hour L2 figure is the *uncontended* CPU d02 cost and is the fair denominator.

---

## 5. Does a clean standalone-d02 CPU benchmark exist, or is one needed?

- **A clean denominator already exists** in the corpus: the **2-domain L2 runs** give a stable, reproducible CPU d02 cost (80–86 s/fc-hour clean compute) across two independent days, on the same WRF V4.7.1 / 28-rank / 160×67×45 / dt=6s config the GPU targets. This is sufficient to publish ~5× with provenance.
- **A still-cleaner number would require a new run** that does NOT exist: a **single-domain (max_dom=1) standalone d02** CPU-WRF run, uncontended, to remove the residual d01-parent coupling and confirm the 83 s/fc-hour. This is the only thing that would tighten the band — but it is a *new WRF run* and is out of scope for this READ-ONLY task. **Recommendation:** if the paper wants to defend the exact 5.x×, queue one uncontended `max_dom=1` d02 CPU-WRF run (cores 4–31, no GPU job, no other nest) and re-measure. Otherwise the existing L2 evidence supports "~5× (band 5–8×)" honestly.

---

## Provenance manifest

- GPU numerator: `proofs/perf/segscan_24h.json` (15.35 s/fc-hr, dt=10s, fp64, RTX 5090; tracked commit `177c734`), `proofs/perf/warmed_timing.json` (16.39 s/fc-hr).
- CPU denominator A: `/mnt/data/canairy_meteo/runs/wrf_l2/20260528_18z_l2_72h_20260529T002423Z/rsl.error.0000` + `namelist.input` (WRF V4.7.1, 28 ranks, d02 3km 160×67, dt=6s, 43200 d02 steps, median 0.1431 s/step).
- CPU denominator B: `/mnt/data/canairy_meteo/runs/wrf_l2/20260527_18z_l2_72h_20260528T002306Z/rsl.error.0000` (median 0.1338 s/step).
- Contended reference: `/mnt/data/canairy_meteo/runs/wrf_l3/20260529_18z_l3_24h_20260530T054804Z/` (live 5-domain run, read-only).
