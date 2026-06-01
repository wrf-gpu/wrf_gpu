# Post-fix consolidation — both headline fixes hold TOGETHER on the REAL operational product path; equivalence re-measured

Date: 2026-06-01
Agent: Opus 4.8 MAX (worker/opus/final-verdict, main working tree)
HEAD: `75b9b40` = theta-limiter-removed (`13dbe4f`/`512a40e`, operational_mode.py)
  + base-`alb`-from-phb-inversion (`6d284ba`/`75b9b40`, acoustic_wrf.py + d02_replay.py)
Owned: validation scripts + new proof objects only (NO production src/ edits).

## TL;DR (continuous-proof checkpoint — VERDICT)

Both fixes were previously validated on PARTIALLY DIFFERENT harnesses (limiter via the
full production wrfout path → 1.52 K; alb-gate via the OOM-safe segmented `_advance_chunk`
→ 0.63 K). This checkpoint measures the **CLEAN COMBINED** numbers on the SAME real
operational product path (the wrfout-writing `execute_daily_pipeline` →
`run_forecast_operational` path the product + d03_replay use). **Both fixes HOLD
TOGETHER, nothing regressed:**

| gate | result | combined-fix headline |
|---|---|---|
| 1. idealized 6/6 | **PASS** (bit-identical) | dycore protected by both fixes |
| 2. d03 1km 24h (product path) | **PASS** D03_1KM_VALIDATED | T2 RMSE mean 1.12 K, psfc bias **-355 Pa** (+2.6 kPa GONE) |
| 3. d02 case3 24h (product path) | **PASS** | T2 RMSE **0.644 K**, bias +0.05 K, psfc -360 Pa |
| 3. d02 case2 72h (product path) | **PASS**, finite to 72h | T2 RMSE **0.623 K**, bias +0.45 K, psfc -481 Pa |
| 4. TOST n=3 MAM (post-fix) | NOT_EQUIVALENT_OR_UNDERPOWERED | T2 ΔRMSE +0.863→**+0.734 K** (still 3.4× over margin) |
| 5. d02 mid-lead PBL residual | REAL, diurnal, daytime-land | THonly peaks +0.9–1.6 K at midday leads (HFX over-flux) |

The two headline fixes together collapse the d02 product T2 RMSE from the raw-product
3.78 K → **0.62–0.64 K** and eliminate the +2.6 kPa pressure-Exner artifact on BOTH
domains. What REMAINS is the daytime PBL theta-side over-flux (the MYNN/HFX debt,
V0.2.0 P1-4a) — small in the corpus-vs-corpus T2 RMSE (because Ponly partly cancels it)
but it keeps the station-paired TOST T2 outside the frozen margin.

---

## Gate 1 — idealized re-confirm (PASS, bit-identical)

`pytest tests/idealized/test_dycore_close_gate.py -m close_gate` through the unified
operational dycore (`_physics_boundary_step`, the step `run_forecast_operational` calls)
on combined-fix HEAD. **6/6 checks PASS**, identical to baseline:
- warm bubble: thermal_rise 1924.35 m, theta' max 1.92 K, max|w| 11.68, mass drift 0.
- Straka: front 14150 m, 4 rotors, theta' min -9.971, max|w| 14.575, mass drift 2.25e-9.
Both fixes are strict no-ops on the doubly-periodic neutral-300 idealized base.
Proof: `proofs/sprintU/close_gate/{warm_bubble,density_current}_verdict.json`.

## Gate 2 — d03 1km 24h on the FULL operational product path (PASS)

`scripts/d03_replay.py --run-id 20260521 --hours 24` → `execute_daily_pipeline` →
per-hour `run_forecast_operational` (force_geopotential=False nested, M9 surface
diagnostics, hourly land refresh). **D03_1KM_VALIDATED**, all_finite, 24/24 wrfouts,
stable 24h, wall 1754s (73s/fh). Scored vs corpus L3 d03 (1km Tenerife) truth:

- **T2 RMSE**: mean **1.12 K**, every lead < 3.0 K (6h 0.88, 12h ~1.51, 24h 1.15);
  beats persistence on T2 at **ALL 24 leads**. T2 bias small/oscillating around 0
  (mean -0.27 K; was +1.27 K pre-fix).
- **PSFC bias**: mean **-355 Pa** (final -364, hour1 -230) — the +2.6 kPa drift is GONE
  (was +2656 Pa pre-fix). Ponly -0.30 K (was +2.0 K), THonly +0.12 K. → the 6h 0.72 K
  result HOLDS to 24h.
- **U10/V10**: RMSE under the 7.5 gate every lead (max V10 4.28); beat persistence
  early/late, lose mid-leads (overnight) — within gate.
Proofs: `proofs/v010_validation/d03_{validation,summary,pipeline_run...}_postfix24h.json`.

## Gate 3 — d02 3km on the FULL operational wrfout path (PASS, both cases)

**Path note (infrastructure, NOT a fix regression):** the production single-scan
`run_forecast_operational` needs ONE 16 GiB XLA intermediate per forecast hour on the
66×159 (Grid B) d02 grid → CUDA_ERROR_OUT_OF_MEMORY at MEM_FRACTION 0.80, exactly the
limit the pressure-drift agent documented (the first attempt confirmed it: PIPELINE_BLOCKED
RESOURCE_EXHAUSTED). Resolved by routing the SAME product path through
`run_forecast_operational_segmented(segment_steps=60)` — **BITWISE identical** to the
single scan (`proofs/perf/segscan_equiv.json`: max abs diff == 0 on every field incl.
the radiation step) but memory-bounded. This still exercises every operational operator
including the `_refresh_grid_p_from_finished → diagnose_pressure_al_alt` alb-from-phb
pressure fix and the de-load-beared theta limiter. Driver:
`proofs/v010_validation/d02_oomsafe_production_run.py` (new validation script).

| case | grid | hours | all_finite | wrfouts | wall | T2 RMSE (mean) | T2 bias | psfc bias | Ponly | THonly |
|---|---|---|---|---|---|---|---|---|---|---|
| case3 (20260521 L3) | B 66×159 | 24 | ✅ | 24/24 | 831s | **0.644 K** | +0.054 K | -360 Pa | -0.30 K | +0.57 K |
| case2 (20260509 L2) | A 66×120 | 72 | ✅ | 72/72 | 1492s | **0.623 K** | +0.446 K | -481 Pa | -0.40 K | +1.09 K |

Pre-fix → limiter-only (512a40e) → **COMBINED (HEAD)** on case3:
T2 RMSE 3.78 → 1.52 → **0.644 K**; T2 bias +3.44 → +1.31 → **+0.054 K**;
psfc +2456 → +2456 → **-360 Pa**; Ponly ~+2.0 → +2.01 → **-0.30 K**.
case2 confirms STABLE TO 72h (winds U10 max 3.99, V10 max 2.93 — within the 5–9 gate).
Note: case1 (20260529) d02 corpus was PURGED → not runnable; case2+case3 are the
available intact-corpus MAM days.
Proofs: `proofs/v010_validation/d02_t2bias_diag_{case3,case2_L2}_COMBINED.json`,
`proofs/v010_validation/pipeline_run_d02_oomsafe_postfix_{case3,case2_L2}.json`.

## Gate 4 — equivalence re-measure (TOST), post-fix (HONEST: NOT EQUIVALENT / UNDERPOWERED)

Re-ran the ADR-029 station-paired TOST (`proofs/m20/tost_ensemble_runner.py`) on HEAD,
3 INDEPENDENT MAM days with intact d02 corpus: 05-09 (case2_L2, 24-72h block),
05-21 (case3_L3, 0-24h), 05-30 (case4_0530_L3, 0-24h) — the fresh 05-30 day REPLACES
the purged 05-29. **SAME harness** (`_advance_chunk` disable_guards=True, station-paired
vs AEMET, complete-pair deletion, ≥30-pair-per-block floor) as the frozen pre-fix run.
Written to a **NEW** file `proofs/m20/tost_run/tost_postfix.json` (+ `postfix/` subdir);
the frozen `proofs/m20/tost_run/tost_aggregate.json` is **UNCHANGED**
(md5 c12427318eacfd76672e29c22d1d40f6 verified).

Paired ΔRMSE = RMSE_GPU − RMSE_CPU vs AEMET, frozen ADR-029 ±margins (§3 of the
Wave−1 manifest, verbatim: T2 ±0.2149 K, U10 ±0.2306 m/s, V10 ±0.2752 m/s):

| var | prefix mean Δ | **postfix mean Δ** | margin | within (point)? | TOST equiv? | ×over margin pre→post |
|---|---|---|---|---|---|---|
| **T2** | +0.863 K | **+0.734 K** | ±0.215 | **NO** | NO | 4.0× → 3.4× |
| U10 | +0.095 | +0.184 m/s | ±0.231 | **YES** | NO (underpowered) | 0.4× → 0.8× |
| V10 | +0.055 | +0.253 m/s | ±0.275 | **YES** | NO (underpowered) | 0.2× → 0.9× |

Per-day deltas T2: [0.635, 0.908, 0.660]. Per-block T2 (GPU vs CPU RMSE vs AEMET):
case2_L2 24-48h Δ+0.484 (gpu 2.49/cpu 2.01), 48-72h Δ+0.787 (2.86/2.07);
case3_L3 0-24h Δ+0.908 (3.06/2.15); case4_0530 0-24h Δ+0.660 (2.90/2.24).

**Honest verdict (preserved, not amended): NOT_EQUIVALENT_OR_UNDERPOWERED.**
- The fixes IMPROVED the station-paired T2 delta (+0.863 → +0.734 K) and removed the
  +2.6 kPa pressure artifact, but **T2 is STILL OUTSIDE the frozen margin** — the GPU
  runs ~0.7 K worse than CPU-WRF against real AEMET stations. T2 fails by **BIAS, not
  power** (the residual daytime PBL over-flux, Gate 5). **Backfilling n cannot fix T2**;
  closing the daytime MYNN/HFX over-flux (V0.2.0 Wave-1 P1-4a) is the precondition.
- U10 (+0.184) and V10 (+0.253) means are now WITHIN their margins by point estimate,
  but TOST is NOT formally equivalent at n=3 (CI upper bound crosses the margin →
  underpowered, exactly the ADR-029 n≥15 floor reason). Note the post-fix wind deltas
  are slightly LARGER than pre-fix (winds are not what the theta/pressure fixes target;
  the larger 05-30 day raises the wind delta), so the n=3 CI is wider.
- **SINGLE-SEASON MAM, n=3 independent days — UNDERPOWERED for a seasonal claim; NEVER
  "seasonal."** ADR-029 floor n≥15, target n≈27.

## Gate 5 — d02 mid-lead PBL residual: REAL after both fixes, diurnal, daytime-land HFX

The pre-fix "12h ≈ 1.35 K theta-side" was a mixed signal. After BOTH fixes the pure
theta-side T2 component (THonly = swap GPU theta into corpus pressure) is **REAL and
persists**, with a clear DIURNAL signature peaking at the local-midday leads:

| lead | case3 THonly | case2 THonly |
|---|---|---|
| 6h (early eve) | +0.49 K | +0.58 K |
| **12h (midday)** | **+0.91 K** | **+1.01 K** |
| 15h | +0.95 K | +1.03 K |
| 24h | +0.21 K | +0.72 K |
| 36/60h (case2 next middays) | — | +1.32 / +1.65 K |

Land/sea/day/night budget (GPU − corpus) localizes it to **DAYTIME LAND**:
- case3 land_day: T2 +0.90 K, **HFX +184 W/m²**, LH +160, THonly +0.78 K.
- case2 land_day: T2 +1.46 K, HFX +107, LH +132, THonly +1.63 K.
- night & sea: much smaller (case3 land_night THonly +0.37, sea_day +0.55).

→ This is the **GPU daytime sensible-heat OVER-FLUX** warming the surface layer at
midday — the same MYNN/HFX surface-flux debt flagged for v0.1.0 d03 and scheduled as
V0.2.0 Wave-1 **P1-4a**. It is NOT the +2.6 kPa pressure artifact (now Ponly -0.3 to
-0.4 K, opposite sign) and NOT the load-bearing limiter (removed). In the corpus-vs-
corpus T2 RMSE this residual is partly MASKED because Ponly (-0.30 K, slight cool)
partly cancels THonly (+0.9 K warm) → net T2 bias at 12h is only +0.35–0.44 K. But
against real AEMET point obs the daytime warm bias is what holds the TOST T2 delta
(+0.734 K) outside the margin. **CHARACTERIZED ONLY — not fixed (out of scope; P1-4a).**

---

## What this checkpoint establishes (and does NOT)

- ESTABLISHED: both headline fixes hold TOGETHER on the real wrfout-writing product
  path; the dycore is protected (idealized 6/6); the +2.6 kPa pressure-Exner artifact
  is GONE on d02 AND d03; the d02 product T2 RMSE = 0.62–0.64 K (was 3.78 K raw product);
  d02 stable to 72h, d03 to 24h, all finite; nothing was forced or masked.
- NOT ESTABLISHED (honest): full TOST equivalence. T2 still fails the frozen margin by
  the residual daytime PBL over-flux (P1-4a precondition); U10/V10 within margin but
  underpowered at n=3, single-season MAM only.

## Files / proofs (all on HEAD, committed)
- `proofs/sprintU/close_gate/*` — idealized 6/6 (Gate 1).
- `proofs/v010_validation/d03_{validation,summary}_postfix24h.json`,
  `pipeline_run_d03_postfix24h.json` — d03 24h (Gate 2).
- `proofs/v010_validation/d02_oomsafe_production_run.py` — OOM-safe product driver (new).
- `proofs/v010_validation/d02_t2bias_diag_{case3,case2_L2}_COMBINED.json`,
  `pipeline_run_d02_oomsafe_postfix_{case3,case2_L2}.json` — d02 (Gates 3, 5).
- `proofs/m20/tost_postfix_manifest.json`, `proofs/m20/tost_run/tost_postfix.json`,
  `proofs/m20/tost_run/postfix/*` — TOST re-measure (Gate 4). Frozen
  `proofs/m20/tost_run/tost_aggregate.json` UNCHANGED.
