# v0.9.0 CORE Validation + Benchmark Burst

**Lane:** worker/opus/v090-validation-burst (branched from worker/opus/v090-release-trunk @ 2162e04, the merged 7-branch trunk)
**Date:** 2026-06-04
**Mode:** WRF-faithful, ADR-007 gated-fp32 (theta/u/v/qv fp32; mu/p/ph/w + acoustic/pressure accumulators fp64) — the OPERATIONAL SHIP mode.
**Resource:** GPU lock claimed (preserving cpu_cores_4_31 backfill); orchestration pinned to cores 0-3; ONE GPU job at a time.
**Honesty policy:** report real skill numbers; do not inflate.

---

## Objective

Close the two genuinely-open 0.9.0 gates and fill the honest speedup:
- **(A)** d02 multi-hour coupled SKILL vs CPU-WRF (only finiteness was previously confirmed).
- **(B)** d03 1km validation with the new faithful physics.
- **(3)** Fill the real-user-time speedup benchmark (9/3km nested + 1km).

The d02-replay hour-1 blow-up is FIXED in the merged trunk (validated stability namelist
epssm=0.5/damp_opt=3/w_damping=1/diff_6th_opt=2/zdamp=5000/dampcoef=0.2/top_lid=True via
fix-B, plus the MYNN qke cold-start seed via qkefix-followup). d02 finite through 3h and d03
finite at 24h were already confirmed pre-burst.

---

## Precision-mode note (why a harness change was needed)

The merged-trunk d02/d03 replay scripts hardcoded `force_fp64=True`. The 0.9.0 SHIP mode is
ADR-007 gated-fp32. Added a `--gated-fp32` CLI flag to both `scripts/m7_l2_d02_replay.py` and
`scripts/d03_replay.py` (default stays full-fp64; flag flips a module `_FORCE_FP64` that the
case-builder reads, since `execute_daily_pipeline` fixes the case_builder signature). Verified
the gated matrix in `src/gpuwrf/contracts/precision.py` (theta/u/v/qv/q*/qke = FP32_GATED;
mu/p/ph/w/ustar/fluxes = FP64) matches ADR-007.

## Corpus / reference setup

- d02 CPU truth: backfilled 28-rank CPU-WRF v4.7.1 L2 d02 wrfout in
  `/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/`. These dirs are wrfout-only (the
  matching `wrf_l2/<run_id>` dirs have `namelist.input` but the wrfout was purged — that is why
  the backfill regenerated them). Built a composite staging dir `/tmp/vburst_runs/<run_id>` that
  symlinks the backfill wrfout + the matching corpus `namelist.input`/`wrfinput` (the loader reads
  grid metadata from the namelist). Honest: real CPU-WRF wrfout + the real namelist that produced it.
- Selected d02 case: **20260507_18z_l2_72h_20260513T124307Z** — representative stable mid-season
  (MAM) case, d02 mass grid 66x159 (the canonical L2 d02 grid; some backfill runs are a smaller
  66x120 sub-domain and were excluded).
- d03 CPU truth: `/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z`
  (intact: namelist + 25 d02 + 25 d03 wrfout; d03 1km mass grid 75x93, the Tenerife domain).

---

## PART 1 — d02 coupled multi-hour SKILL vs CPU-WRF

### 1h gated-fp32 smoke (path confirmation)
- Verdict **L2_D02_GREEN**, all statuses PASS, finite throughout.
- Final-hour Tier-4 RMSE vs CPU-WRF: **T2 0.492 K (bias -0.315), U10 0.427 m/s (-0.155), V10 0.325 m/s (-0.008)** — well inside bars (T2<3.0, U10/V10<7.5).
- Wall: total 561.8 s; hour-1 (compile-inclusive) 486.7 s.

### Multi-hour (24h) gated-fp32 — coupled skill (proofs/v090/d02_coupled_skill.json)
- **FINITE + bounded all 24 hours** (theta 290-498 K; |u|<49, |v|<15, |w|<2.7 m/s). bounds PASS. No blow-up.
- Per-lead RMSE / bias vs CPU-WRF (mean over 24 leads / final-24h):
  | field | mean RMSE | final RMSE | mean bias | bar | verdict |
  |---|---|---|---|---|---|
  | T2 | 1.11 K | 1.30 K | -0.30 K | 3.0 K | within bar EVERY lead |
  | V10 | 3.57 m/s | 4.26 m/s | -2.43 | 7.5 | within bar EVERY lead |
  | U10 | 4.41 m/s | 8.04 m/s | -3.61 | 7.5 | within bar 20/24 leads; crosses ~22h |
  | HFX | 63 W/m² | 43 W/m² | +2.0 | 120 (info) | within band |
  | PBLH | 151 m | 177 m | +18 m | 400 (info) | within band |
- **U10 grows monotonically 0.43 (1h) -> 8.04 (24h) m/s**, a systematic weak-wind bias (GPU ~7 m/s too weak by 24h). This is the documented near-surface U-momentum bias, now quantified at d02 multi-hour scale. U10 still **beats persistence at 23/24 leads**; V10 beats persistence 23/24.
- **Verdict (24h):** within operational margins for T2/V10/HFX/PBLH at every lead and for U10 through ~21h; the only binding-bar breach is U10 at the last ~3 leads. Stable, physical, no blow-up.
- Wall (24h, gated-fp32, COLD radiation-graph compile): total 1166.5 s; hour-1 67.1 s, hour-2 435.6 s (cold RRTMG+physics+dycore compile), steady-state median 27.75 s/fc-hr.

### Full-horizon 72h gated-fp32 (proofs/v090/d02_coupled_skill_72h.json + speedup_d02_72h) — **L2_D02_GREEN**
- **FINITE + bounded all 72 hours; final-hour Tier-4 RMSE PASS: T2 0.81 K, U10 4.00 m/s, V10 2.97 m/s** (all within bars).
- Per-lead over 72h: **T2 within bar 72/72** (mean 1.06 K), **V10 within bar 72/72** (mean 3.21), **U10 within bar 66/72** (mean 4.79, max 8.04). **U10 beats persistence 71/72.**
- **The U10 weak-wind bias is DIURNAL/EPISODIC, not a runaway:** U10 RMSE rises to ~8.0 around the hour-22-30 peak-wind window then RECOVERS to 3-4 m/s by 72h (6h-cadence: 1.7, 4.3, 6.6, **8.0**, 6.7, 6.5, 5.9, 5.4, 3.6, 2.9, 3.1, 4.0). The 24h-endpoint number (8.04) caught the diurnal peak; the 72h endpoint (4.00) is well within bar. This recontextualizes the U10 gap as an episodic peak-wind under-prediction, not a degrading instability.
- Wall (72h, gated-fp32, WARM cache): total 2149.9 s; hour-1 66.8 s, **hour-2 57.3 s** (radiation graph reused from the persistent jax cache the 24h run populated -> operational daily-cadence scenario), steady-state median 27.72 s/fc-hr.

---

## PART 2 — d03 1km validation (new faithful physics) — proofs/v090/d03_1km_validation.json
- **BLOCKED / FAIL in gated-fp32 (the ship mode): NONFINITE after forecast hour 1.**
- **Worst (and only) field: qke** (MYNN turbulent kinetic energy, fp32) — 3036 nonfinite cells after hour 1; every other prognostic field stays finite.
- **Root cause (isolated):** qke is `FP32_GATED` (precision.py:112) and lives OUTSIDE the fp64-locked mass/pressure/acoustic path that keeps d02 stable. At 1km (steep Tenerife terrain + dt=3s) its TKE budget overflows fp32.
- **Full-fp64 d03 IS finite** (prior trunk proof, 0.3h/360 steps) but ~1:64-throttled (~16 min/0.3h) so a 24h fp64 validation cannot complete in the burst window.
- **=> the 1km gate is NOT met in either practical mode today.** No timesteps were produced (NaN'd at hour 1), so no T2/U10/V10/PBLH/precip/prognostic-level RMSE could be computed.
- **Actionable fix (OUT OF this burst's scope — needs a bounded precision-contract sprint + review):** promote `qke` (and likely the MYNN length-scale intermediates) to FP64 in the gated matrix. qke is not a conserved mass/pressure field, so this preserves the gated-fp32 invariants and the d02 speedup (qke is a small field). The `d03_prognostic_pblh_analyze.py` augmentation (PBLH + prognostic-level RMSE) is committed and ready to run the moment a ship-speed d03 completes.

---

## PART 3 — honest real-user-time speedup (proofs/v090/speedup_benchmark.json)
Denominator = 28-rank CPU-WRF d02 own-solver cost (64.6 / 72.0 / 77.3 s/fc-hr conservative/mid/high). All GPU numbers gated-fp32 (ship mode), command-to-finish, compile-INCLUSIVE.
- **9/3km nested (d02), 72h, WARM-cache (operational daily-cadence) real-user headline:** **2.16x** conservative / 2.41x mid / 2.59x high (2149.9 s / 72h = 29.86 s/fc-hr).
- **9/3km nested (d02), 24h COLD-launch (first-ever-run, full RRTMG compile) real-user:** **1.33x** conservative (1166.5 s / 24h). This is the honest worst-case for a never-before-compiled gated-fp32 config.
- Steady-state (compile-EXCLUDED, CONTEXT ONLY): **2.33x / 2.60x / 2.79x** (27.7 s/fc-hr).
- dt-matched strict floor (GPU forced to CPU dt=6s, /1.67): ~0.80x conservative (warm: ~1.29x).
- **1km (d03): UNMEASURED / BLOCKED** — gated-fp32 NaN'd (qke) before any complete wall; full-fp64 too throttled (~16 min/0.3h) to time a 24h run inside the burst. Honestly left a placeholder, marked MEASUREMENT_STATUS=BLOCKED.
- **compile caveat (important):** the d02 forecast traces TWO large XLA graphs — a no-radiation hour graph (~67 s incl ~39 s compile) and a radiation-active hour graph. On a COLD launch the radiation graph costs ~408 s (hour-2 = 435.6 s in the 24h run). JAX persistent compilation cache is ON by default (`gpuwrf.runtime.jax_cache` -> `/mnt/data/gpuwrf_jax_cache`); the 72h run reused it (hour-2 = 57.3 s), which is the realistic daily-cadence wall. Report 2.16x (warm/operational) AND 1.33x (cold first-run) — do not present only one.

---

## Bottom line per gate
- **(A) d02 multi-hour SKILL: GATE MET (with a noted episodic U10 gap).** 72h GREEN, finite/stable throughout, final-hour Tier-4 RMSE PASS (T2 0.81, U10 4.00, V10 2.97). T2/V10 within bar at all 72 leads; U10 within bar 66/72 (episodic diurnal peak-wind under-prediction, recovers; beats persistence 71/72). Within operational margins (v0.1.0/v0.2.0 bars) for the vast majority of the forecast.
- **(B) d03 1km gate: OPEN / FAIL.** gated-fp32 NaNs on qke after hour 1; fp64 finite-but-throttled. Root cause isolated, fix identified (qke->fp64), out of burst scope.
- **(3) speedup: filled honestly.** d02 9/3km real-user 2.16x warm / 1.33x cold; d03 1km blocked.

## Risks / honest gaps
- **d03 1km gate OPEN (qke fp32 NaN)** — the single most important carry-over. Low-risk fix identified (promote qke to FP64; it is outside the mass/pressure path), but it is a precision-contract (`precision.py`) change requiring its own bounded sprint+review, so this burst leaves the 1km gate FAILED, not green.
- **d02 U10 episodic weak-wind bias** — peaks ~8 m/s in the diurnal high-wind window (~h22-30), recovers to ~4 by 72h; within bar 66/72 leads, beats persistence 71/72. Pre-existing documented near-surface U-momentum issue, now quantified; T2/V10/HFX/PBLH unaffected. Not an instability.
- **Cold-launch compile penalty:** first-ever gated-fp32 d02 launch pays a ~7 min radiation-graph XLA compile (24h cold real-user 1.33x). The persistent jax cache (on by default) removes it on subsequent runs (warm 72h 2.16x), which is the operational daily-cadence reality.
- **gated-fp32 = the ship mode** confirmed active in every run (the fp64->fp32 scatter FutureWarning fires every step; precision.py matrix verified: theta/u/v/qv/qke fp32; mu/p/ph/w + acoustic/pressure fp64).
- **Single case / single season:** d02 skill is one representative mid-season (MAM) case (20260507). Not a powered ensemble; consistent with the corpus n-power limits noted elsewhere.

## Files changed / proofs
- scripts/m7_l2_d02_replay.py, scripts/d03_replay.py (--gated-fp32 flag)
- proofs/v090/d02_coupled_skill_analyze.py, proofs/v090/d03_prognostic_pblh_analyze.py
- proofs/v090/d02_coupled_skill.json, proofs/v090/d03_1km_validation.json,
  proofs/v090/d03_prognostic_pblh.json, proofs/v090/speedup_benchmark.json (filled)
