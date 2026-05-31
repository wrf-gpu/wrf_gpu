# v0.1.0 Seasonal TOST Equivalence Gate — Feasibility Verdict & Campaign Plan

**Date:** 2026-05-31
**Author:** Opus worker (`worker/opus/final-verdict`), CPU-only feasibility assessment
**Scope:** Proof-table row 6 — paired TOST equivalence (T2/U10/V10) of the GPU model
vs CPU-WRF v4, on a **≥15-case (target ≥30) seasonal** ensemble (ADR-029 / reset plan).
**Constraint:** REUSE-ONLY (no new WRF runs unless explicitly approved).

---

## 0. Bottom line (decision-critical)

**A ≥15-case SEASONAL TOST is NOT reuse-only-feasible.** The parallel worker's finding is
**CONFIRMED and refined**:

- The corpus retains **3 distinct usable forecast DAYS** with complete d02 history, all
  **MAM (spring 2026)** — fewer than the "5 cases" headline once L2/L3 siblings of the same
  init day are correctly counted as one (non-independent) synoptic situation.
- ADR-029 requires **n ≥ 15 (target n ≈ 27/30)** of statistically independent cases, across
  **multiple seasons**. Reuse-only gives **n = 3 days, 1 season**. The gap is ~5× on count
  and total on seasonal breadth.
- This is a **logistical** shortfall, not a fundamental one: the harness, scorer, and margins
  are validated and ready (see §3–4); the corpus is the only missing piece.

**Recommended path:** **Option B + staged Option A.** Run the *plumbing-and-margins TOST
now on the achievable n=3 (MAM)*, report it as a **within-corpus spring-trade-wind
equivalence check (v0.1.0, explicitly underpowered + single-season)**, and in parallel begin
the **CPU-WRF May backfill (Option A)** to reach n≈15 MAM for a defensible single-season TOST,
while enrolling a permanent nightly retained-output validation case to grow JJA/SON/DJF for a
*true seasonal* claim at v0.2.0. **Do NOT label any single-season result "seasonal."**

---

## 1. Honest usable-case inventory (verified on disk 2026-05-31, not from stale manifest)

The `cpu_baseline_manifest.json` (generated 2026-05-29) lists 5 cases; **two of those
(both 20260528) have since been purged** by the nightly pipeline and replaced by 20260529.
Physical disk inspection (count of `wrfout_d02` frames per run dir) gives the current truth:

### 1a. Complete-series runs on disk NOW

| run-id | init date | season | level | domains | fhours | d02 frames | complete? | usable for paired d02-TOST? |
|---|---|---|---|---|---|---|---|---|
| `20260509_18z_l2_72h_20260511T190519Z` | 2026-05-09 | **MAM** | L2 | d01,d02 | 72 | 73/73 | yes | **partial** — obs start 05-11 09z, so only the 48–72h lead block scores |
| `20260529_18z_l2_72h_20260530T054804Z` | 2026-05-29 | **MAM** | L2 | d01,d02 | 72 | 73/73 | yes | **yes** — 0–24/24–48/48–72h all obs-covered (obs to 05-31 13z) |
| `20260509_18z_l3_24h_20260511T190519Z` | 2026-05-09 | **MAM** | L3 | d01–d05 | 24 | 25/25 | yes | **no** — 0–24h leads all valid BEFORE obs start (05-10 18z < 05-11 09z) |
| `20260521_18z_l3_24h_20260522T133443Z` | 2026-05-21 | **MAM** | L3 | d01–d05 | 24 | 25/25 | yes | **yes** — fully obs-covered; the ADR-029 benchmark-anchor case |
| `20260529_18z_l3_24h_20260530T054804Z` | 2026-05-29 | **MAM** | L3 | d01–d05 | 24 | 25/25 | yes | **yes** — fully obs-covered |

Partial/leftover dirs with a handful of frames (`20260428` n=3, `20260509…T154354Z` n=3,
`20260521…T072630Z` n=9, `20260521_l2` n=20, `20260530` n=2) are **not complete forecasts**
and are excluded.

### 1b. Usable-case count BY SEASON (the answer)

| Season | Distinct usable forecast DAYS | Notes |
|---|---|---|
| **DJF** | 0 | none |
| **MAM** | **3** (05-09, 05-21, 05-29) | the entire corpus |
| **JJA** | 0 | none |
| **SON** | 0 | none |

- **Distinct independent usable cases = 3** (05-09 only via its later lead block; 05-21 and
  05-29 fully). The 5 "scoring units" the harness enumerates (case1_L2, case1_L3, case2_L2,
  case2_L3, case3_L3) are **NOT 5 independent samples**: each init day's L2 and L3 share the
  same synoptic situation and overlapping d02 grid, so for TOST `n` they count as the day, not
  the unit. Counting units inflates the apparent n and must not be used for the power claim.

### 1c. Why 66 dirs collapse to 3 usable days (the precise gap)

1. **Output purge (61 → 5).** The Gen2 nightly keeps `wrfout` for only ~2 days; 61 of 66 run
   dirs have had their `wrfout` deleted (only `wrfinput`/`wrfbdy`/`namelist.input` survive).
   → 5 dirs retain complete d02 frames.
2. **L2/L3 sibling de-duplication (5 → 3 days).** The 5 surviving complete dirs are only
   **3 distinct init days** (05-09 has both L2+L3; 05-29 has both L2+L3; 05-21 has L3 only).
   Independent TOST samples = days, not run dirs → **3**.
3. **Obs-window clipping (3 → effectively 2 full + 1 partial).** Hourly AEMET obs begin
   2026-05-11 09z. The 05-09 L3 (24h) ends 05-10 18z — entirely before obs — so it is
   **unscoreable**; the 05-09 L2 (72h) only overlaps obs in its 48–72h block. Fully scoreable
   days = **2** (05-21, 05-29); partially = **1** (05-09 L2, late block only).
4. **Single season.** All 66 dirs — surviving or purged — are late-Apr→late-May 2026 = **MAM**.

---

## 2. TOST design vs the achievable corpus

From `tost_design.json` + `paired_tost_scorer.py` (predeclared, ADR-029):

| Variable | CPU-WRF RMSE benchmark | Equivalence margin (10%) | provisional σ (20%) | MDE@n15 | MDE@n30 | **required n for 10% MDE** |
|---|---|---|---|---|---|---|
| **T2** | 2.1487 K | 0.2149 K | 0.4297 | 0.2917 | 0.2003 | **27** |
| **U10** | 2.3065 m/s | 0.2306 m/s | 0.4613 | 0.3132 | 0.2150 | **27** |
| **V10** | 2.7523 m/s | 0.2752 m/s | 0.5505 | 0.3737 | 0.2566 | **27** |

- Test: paired TOST on case-level RMSE deltas (RMSE_GPU − RMSE_CPU); equivalence accepted
  only if BOTH one-sided tests reject at α=0.05 for **every** variable.
- α=0.05, β=0.2 (power 0.8); margin = 10% of CPU-WRF RMSE; per-block exclusion if <30 pairs.
- **Does the usable corpus meet it? NO.** n=3 days « n=15 floor « n≈27 target. At n=3 the
  test has ~no power: it can only return "equivalent" if the GPU/CPU delta is near-zero with
  near-zero variance, and the predeclared per-variable MDE (0.20–0.27) is not even computed at
  n<15. ADR-029 itself flags n=15 as underpowered for the 10% MDE; n=3 is far below that.
- **Validated cross-check:** the scorer reproduces the ADR-029 CPU-WRF RMSE benchmarks
  **exactly** (T2 2.1487 / U10 2.3065 / V10 2.7523) on the 05-21 L3 case — confirming the
  margins, obs join, and scorer are mutually consistent (CPU-vs-CPU self-test = 0.00 delta).

---

## 3. The ready-to-launch harness

**`proofs/m20/tost_ensemble_runner.py`** — CPU-orchestrated, GPU-sequenced. Built and
**validated CPU-only end-to-end** (the GPU forecast step is the only un-exercised link, by
design — the manager runs it sequenced on GPU after the HFX+precip fixes).

What it does, reusing existing validated code (no model code touched):
1. **GPU forecast** per (case, level) via the SAME carry-advance loop as
   `proofs/v010_validation/v010_d02_validate.py` (`_build_real_case` + `_advance_chunk`),
   advancing ONE carry to max scoreable lead (bounded memory, `block_until_ready` per segment).
2. **GPU wrfout emit** — writes GPU T2/U10/V10 (+XLAT/XLONG) per obs-covered lead into a
   minimal NetCDF `wrfout_d02_<valid>` the frozen M7 station interpolator reads unchanged.
3. **Station-paired TOST** via `proofs/m20/paired_tost_scorer.py` — CPU/GPU/obs complete-pair
   join, per-case paired delta RMSE, ADR-029 predeclared-margin TOST, aggregate verdict.

**Launch commands (manager, sequenced on GPU after physics fixes):**
```bash
# dry plan (CPU only):
PYTHONPATH=src taskset -c 0-3 python proofs/m20/tost_ensemble_runner.py --plan

# full GPU run + aggregate TOST (default manifest = 3 MAM days):
PYTHONPATH=src OMP_NUM_THREADS=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 \
  python proofs/m20/tost_ensemble_runner.py \
    --out-dir proofs/m20/tost_run --execute

# after backfill: swap --manifest proofs/m20/tost_corpus_manifest.json (n>=15)
```

**Plumbing proof:** a CPU-only self-test (emit GPU-wrfout = copy of CPU truth → score)
produced paired delta = `+0.00e+00` for all of T2/U10/V10 and reproduced the exact ADR-029
CPU benchmarks. The full chain is correct; only the real GPU forecast remains to be run.

### Per-case GPU runtime + total GPU-hours for the achievable n

From `v010_d02_result.json` (same advance loop, measured on this RTX 5090):

| unit | forecast | measured GPU wall |
|---|---|---|
| L2 72h (full, 5 leads) | 72h | ~1410 s case1 / ~1220 s case2 → **~20–24 min** |
| L3 24h (full, 3 leads) | 24h | ~590–600 s → **~10 min** |

The TOST harness scores **every obs-covered integer lead hour** (not 5 sparse leads), so it
snapshots more frequently but advances the **same single carry** — wall is dominated by the
integration, so the per-case figures above are the right order (snapshotting adds only
diagnostic+I/O, a few seconds per lead).

- **Achievable n=3 corpus GPU cost:** 05-29 L2 (24 min) + 05-29 L3 (10 min) + 05-21 L3
  (10 min) + 05-09 L2 (20 min) + (05-09 L3 unscoreable) ≈ **~64 min ≈ 1.1 GPU-hours**, one
  sequenced GPU slot.
- **Backfilled n=15 (MAM, L2 72h):** 15 × ~22 min ≈ **~5.5 GPU-hours**.
- **Backfilled n=30 (MAM, L2 72h):** 30 × ~22 min ≈ **~11 GPU-hours**.

(GPU cost is small and not the binding constraint; the CPU-WRF truth backfill is — see §4.)

---

## 4. Options to close the gap

### Option A — Generate more CPU-WRF cases (backfill, no AIFS fetch needed)

The raw AIFS GRIB2 forcing is **preserved for 35 init dates** in
`/mnt/data/canairy_meteo/runs/forcing_cases/` (30 of them May 2026). Backfill needs only the
local `WPS(metgrid) → real.exe → wrf.exe` chain — **no network/data dependency**. CPU-WRF runs
on **28 ranks (cores 4–31)**, so they do **not** contend with the GPU.

**CPU cost per case (measured from rsl logs on this box):**

| run type | config | measured CPU wall (28-rank) | source |
|---|---|---|---|
| **L2 72h** | max_dom=2 (9/3 km), e_we 94×60×45L, dt=18s | **~5.2 h** (wrfout first→last mtime span 18 770 s; Σ "Timing for main" all-domain = 17 416 s = 4.84 h pure integration) | `20260529_18z_l2_72h` rsl.error.0000 + wrfout mtimes |
| **L3 24h** | max_dom=5 (9/3/1/1/1 km), dt=18s | **~17.7 h** (wrfout first→last mtime span 63 732 s; the Σ per-domain "Timing for main" = 40.6 h is an overcount — parent-domain timing lines include nested-child integration) | `20260529_18z_l3_24h` rsl.error.0000 + wrfout mtimes |

> **Runtime correction (verified by 2nd Opus pass 2026-05-31):** an earlier draft of this table listed L2 72h as ~9.1 h; the authoritative wallclock is **~5.2 h** (first→last wrfout mtime span = 18 770 s; all-domain "Timing for main" sum = 17 416 s of pure integration). The 9.1 h figure was an mis-sum. The L3 ~17.7 h figure is confirmed (the 40.6 h "Timing for main" sum double-counts because WRF parent-domain timing includes nested-child time). **Net effect on Option A: the L2 backfill is CHEAPER than stated — re-tabulated below.**

- **L2 72h is the right backfill target**: one run yields all three lead blocks
  (0–24/24–48/48–72h) and costs ~9 h CPU vs ~18 h for the 5-nest L3. The 9 km/3 km d02 is the
  scoring domain for both, so L2 is strictly cheaper per scoreable case.
- **Obs-overlapping forcing dates available for backfill:** **20 dates** with init ≥ 2026-05-11
  (fully obs-covered leads) + 3 partial (05-08…05-10). Enough to reach **n≈15–20 MAM**.

**Backfill cost to targets (L2 72h, serial on the 28-rank box):**

| target n (MAM) | new CPU-WRF runs needed | CPU wall (serial, ~5.2 h/run corrected) | calendar |
|---|---|---|---|
| n=15 | ~13 (have 2 full + 1 partial) | ~13 × 5.2 h ≈ **68 CPU-h** | ~3 nights of background compute |
| n=20 | ~18 | ~18 × 5.2 h ≈ **94 CPU-h** | ~4 nights |
| n=21 (all preserved obs-overlap dates) | ~19 | ~19 × 5.2 h ≈ **99 CPU-h** | ~4–5 nights |
| n=27 (10% MDE) | (only ~21 May dates exist) → **not reachable from May forcing alone** | — | needs >1 season anyway |

- **Disk:** ~1.8 GB per L2 72h run; `/mnt/data` has ~223 GB free (92% used). A 15-case L2
  corpus ≈ 27 GB — feasible but tight; **thin scored wrfout to T2/U10/V10/RAINNC + XLAT/XLONG**
  or move to a compressed archive promptly, and **patch the nightly purge** to retain enrolled
  validation cases (else the corpus re-depletes and the result is irreproducible).
- **Seasons this fills:** MAM only. JJA/SON/DJF **cannot** be backfilled from preserved forcing
  (no forcing exists for them) — they require going-forward nightly capture.

### Option B — Rigorous within-corpus equivalence on the achievable n (recommended for v0.1.0)

Run the harness now on **n=3 MAM days** and report it **honestly**:

- Headline: *"Within-corpus paired-TOST plumbing + descriptive equivalence check on n=3 spring
  trade-wind cases; the test is PREDECLARED-UNDERPOWERED (ADR-029 needs n≥15/target 27) and
  SINGLE-SEASON (MAM). Reported as a v0.1.0 viability signal, not a seasonal equivalence claim."*
- Emit per-case paired deltas + 90% CIs + empirical σ_v (the harness/scorer already do this).
  The **empirical σ_v from the real GPU deltas** is the key new datum: if σ_v ≤ the 20% planning
  value, the n required to reach the margin may be lower than 27; if it materially exceeds it,
  the corpus must grow further. This measurement is itself worth the n=3 run.
- Requires a **stats reviewer (Opus or agy)** sign-off on the exclusion log (per ADR-029).
- **This is the only path that closes a v0.1.0 row without new WRF runs**, and it is defensible
  *if and only if* it is labeled exactly as underpowered + single-season.

### Option C — Permanent nightly validation case (the real seasonal fix, v0.2.0)

Enroll one fixed L2 72h CPU-WRF + AIFS-forcing case per night into the Gen2 nightly **with
wrfout retained** (thinned to the 5 surface fields + grid). Zero extra human effort; accrues
~30 cases/season. After ~12 months → a genuinely **seasonal ≥30/season** corpus. This is the
only route to a true *seasonal* TOST headline; it cannot be reuse-only because the future
seasons do not exist yet.

---

## 5. Recommended path (for the manager+principal decision)

1. **Now, reuse-only (Option B):** run `tost_ensemble_runner.py` on the n=3 MAM corpus after
   the HFX+precip fixes land (sequenced GPU, ~1.1 GPU-h). Measure empirical σ_v. Report as
   **underpowered + single-season** with stats-reviewer sign-off. Closes proof-table row 6 as a
   *qualified* v0.1.0 viability signal, NOT a seasonal claim.
2. **Approve a bounded CPU backfill (Option A):** if the principal wants n≥15 for v0.1.0,
   re-run ~13 obs-overlapping May L2 72h cases (~68 CPU-h ≈ ~3 background nights on cores
   4–31; does not touch the GPU). Register each in a `tost_corpus_manifest.json` and re-launch
   the same harness. Yields a defensible **single-season (MAM) n≥15 TOST** — still must NOT be
   called "seasonal."
3. **Start Option C immediately** regardless: patch the nightly to retain one enrolled
   validation case/night. This is the only path to a real **seasonal** equivalence claim, and
   it must start now to have JJA/SON/DJF by v0.2.0.

**Hard guardrail (ADR-029):** a single-season corpus, however large, must never be reported as
"seasonal equivalence." The v0.1.0 row should be scoped to *spring trade-wind* conditions; the
seasonal headline is a v0.2.0 deliverable gated on multi-season coverage + reviewer approval.

---

## 6. Artifacts

- Inventory + counts verified on disk 2026-05-31 (this doc §1).
- Harness: `proofs/m20/tost_ensemble_runner.py` (built, CPU-validated end-to-end).
- Scorer: `proofs/m20/paired_tost_scorer.py` (reproduces ADR-029 CPU benchmarks exactly).
- Design/margins: `proofs/m20/tost_design.json`.
- Prior assessment (corroborated, refined): `proofs/m20/seasonal_gap_assessment.md`.
- GPU runtimes: `proofs/v010_validation/v010_d02_result.json`.
- CPU-WRF runtimes: rsl logs under the corresponding `/mnt/data/canairy_meteo/runs/*` dirs.
