# Wave −1 — CPU-WRF Validation-Corpus / TOST Manifest

**Status:** ACTIVE bookkeeping object (v0.2.0 Wave −1, the calendar long-pole).
**Date:** 2026-06-01
**Owner:** Opus 4.8 xhigh (`worker/opus/final-verdict`).
**Scope:** Pure INVENTORY + BACKFILL PLAN over EXISTING artifacts. **No GPU job and no
CPU-WRF run was launched to produce this doc.** Disk truth was read on 2026-06-01; the n=4
TOST aggregate (`proofs/m20/tost_run/tost_aggregate.json`) is frozen and was only read.
**Authoritative spec:** `.agent/decisions/V0.2.0-PLAN.md` (Wave −1 section) + ADR-029
(`.agent/decisions/ADR-029-STATISTICS-DESIGN-TOST.md`). **Margins are FROZEN — see §3.**

This manifest is what makes the future ≥15→27 TOST equivalence gate credible and
**non-gameable**: it pins the case set, the same-grid pairing rule, the complete-pair masks,
the frozen margins, the backfill sequence, and the predeclared exclusion log.

---

## 0. Bottom line (decision-critical)

- **Current independent n = 3 MAM days** (2026-05-09, 2026-05-21, 2026-05-29). A 4th MAM day
  (2026-05-30 L3) has appeared on disk via the nightly and is *near-scoreable* (§1d) but not
  yet enrolled/GPU-paired.
- **The reported "n=4" in `tost_aggregate.json` is 4 SCORING-UNIT lead-block deltas, NOT 4
  independent cases.** Those 4 deltas come from only **3 distinct synoptic days** (case1 L2+L3
  share 05-29; case2 = 05-09; case3 = 05-21). For the ADR-029 power claim, **n = days = 3**.
  Do not advertise n=4. (Counting units inflates n and is forbidden by the §2 power rule.)
- **n-gap: 12 to the n=15 floor; 24 to the n≈27 ten-percent-MDE target.**
- **The corpus is SINGLE-SEASON (MAM / spring trade-wind), 100% of all on-disk and all
  recoverable cases.** Therefore the gate can only ever be a **single-season MAM equivalence
  check**, NEVER "seasonal," until a multi-season corpus is grown going-forward (§4 / Option C).
- **TOP BLOCKER to a credible TOST is NOT sample size — it is the T2 result itself.** On the
  achievable corpus the GPU is **already not equivalent on T2**: mean paired ΔRMSE(T2) =
  **+1.05 K vs a frozen margin of ±0.215 K** (4.9× over). T2 fails by *bias*, not by *power*:
  no amount of backfill makes T2 equivalent at the frozen margin while the GPU runs ~+1 K warm.
  U10/V10 deltas are within margin on this corpus but their tiny empirical σ is from only 3
  correlated days and must not be trusted as the power input (§2). **Closing the warm-T2
  bias (the v0.1.0 HFX/MYNN debt, V0.2.0 Wave 1 item P1-4a) is the real precondition for any
  honest TOST PASS — corpus backfill is necessary but not sufficient.**

---

## 1. Case table (verified on disk 2026-06-01)

### 1a. Pairing model (why cross-grid is allowed)

TOST pairs **GPU vs CPU on that run's OWN d02 grid** (the GPU is initialized from the run's
t=0 `wrfout_d02` and integrated on the same grid; the paired ΔRMSE = RMSE_GPU − RMSE_CPU is
computed over the *identical* complete-pair station rows for that one run). So GPU and CPU are
**always same-grid within a case**. Different cases may sit on different d02 grids; that is
permitted because TOST operates on *case-level paired deltas*, not on a pooled cross-grid
field. The ADR-029 exclusion "exclude a case if its score domain grid differs in shape from
the CPU run" is an **intra-case GPU-vs-CPU** check, which all cases pass by construction.
**Cross-case grid heterogeneity is logged (below) for transparency, not excluded.**

Two distinct d02 grids exist in the corpus (verified via `ncdump -h`):
- **Grid A** `south_north=66 × west_east=120` (e_we=90): the 2026-05-09 case only.
- **Grid B** `south_north=66 × west_east=159` (e_we=94): 05-21, 05-29, 05-30.
Both `dx=9000/3000` 9/3-km Canary nests, 44 levels, same physics suite (CONUS: MP=8 Thompson,
PBL=5 MYNN, sfclay=5, LSM=4 Noah-MP, RRTMG=4, cu=1). Same-physics → poolable as a single
ensemble; the grid split is a footnote, not an exclusion.

### 1b. Complete-series CPU-WRF runs on disk NOW (the usable corpus)

| run-id | init (UTC) | season | level | d02 grid | d02 frames | fhours | obs-cover | usable day? | GPU pair emitted? | TOST status |
|---|---|---|---|---|---|---|---|---|---|---|
| `20260509_18z_l2_72h_…190519Z` | 2026-05-09 18z | MAM | L2 | A 66×120 | 73/73 | 72 | 24–72 h block only (obs start 05-11 09z) | **YES (partial leads)** | YES (`case2_L2`) | **VALIDATED-PAIR** (24-48h, 48-72h blocks) |
| `20260521_18z_l3_24h_…133443Z` | 2026-05-21 18z | MAM | L3 | B 66×159 | 25/25 | 24 | full 0–24 h | **YES** | YES (`case3_L3`) | **VALIDATED-PAIR** (0-24h) — ADR-029 benchmark anchor |
| `20260529_18z_l2_72h_…054804Z` | 2026-05-29 18z | MAM | L2 | B 66×159 | 73/73 | 72 | full 0–72 h | **YES** | YES (`case1_L2`) | **VALIDATED-PAIR** (all 3 blocks) |
| `20260529_18z_l3_24h_…054804Z` | 2026-05-29 18z | MAM | L3 | B 66×159 | 25/25 | 24 | full 0–24 h | sibling of L2 (same day) | YES (`case1_L3`) | **VALIDATED-PAIR** (0-24h) — *not an independent day* |
| `20260509_18z_l3_24h_…190519Z` | 2026-05-09 18z | MAM | L3 | A 66×120 | 25/25 | 24 | none (ends 05-10 18z < obs start) | **NO** | NO (`case2_L3` emitted 0) | **EXCLUDED** (no obs-covered lead) |

### 1c. Independent-day collapse (the honest n)

| Init day | L2 present | L3 present | independent TOST sample? | note |
|---|---|---|---|---|
| 2026-05-09 | ✅ (Grid A) | ✅ (unscoreable) | **1** | scored via L2 24–72 h only |
| 2026-05-21 | partial (20 frames, dropped) | ✅ (Grid B) | **1** | benchmark anchor |
| 2026-05-29 | ✅ (Grid B) | ✅ (Grid B) | **1** | L2+L3 are ONE synoptic day → counts once |
| **Total independent usable days** | | | **3** | all MAM |

**Complete-pair masks (verified, `proofs/m20/tost_run/paired_score_*.json`):**

| unit (init day) | grid | scored lead-blocks (≥30 pairs) | excluded blocks | n_pairs/block (T2) |
|---|---|---|---|---|
| `case1_L2` (05-29) | B | 0-24h, 24-48h, 48-72h | — | 1640 / 1618 / 436 |
| `case1_L3` (05-29) | B | 0-24h | 24-48h, 48-72h (no leads) | 1640 |
| `case2_L2` (05-09) | A | 24-48h, 48-72h | 0-24h (pre-obs) | 0 / 588 / 1080 |
| `case3_L3` (05-21) | B | 0-24h | 24-48h, 48-72h (24 h fcst) | 1639 |
| `case2_L3` (05-09) | A | — | all (no obs-covered lead) | EXCLUDED |

All scored blocks clear the predeclared ≥30-pairs floor by a wide margin (smallest = 436).

### 1d. New since the campaign plan (candidate 4th MAM day — NOT yet enrolled)

| run-id | init | season | grid | d02 frames | obs-cover | status |
|---|---|---|---|---|---|---|
| `20260530_18z_l3_24h_…161057Z` | 2026-05-30 18z | MAM | B 66×159 | 20 (05-30 18z→05-31 13z) | partial (depends on obs end ≥05-31 13z) | **CPU-only — GPU pair NOT yet emitted.** Candidate 4th independent MAM day. Enroll only after a same-grid GPU run + complete-pair check; subject to the nightly purge (capture promptly, §4.1). |
| `20260428_18z_l3_24h_…221139Z` | 2026-04-28 | MAM | (3 frames) | n/a | **EXCLUDED — incomplete (3 frames)** |
| `20260530_18z_l3_24h_…050849Z` | 2026-05-30 | MAM | (2 frames) | n/a | **EXCLUDED — incomplete (2 frames), superseded by the 20-frame rerun** |
| `20260521_18z_l2_72h_…133443Z` | 2026-05-21 | MAM | (20 frames) | partial | **EXCLUDED — incomplete L2 history (20/73); the day is covered by its L3 sibling** |
| `20260509_…T154354Z`, `20260521_…T072630Z`, l2rerun dirs | various | MAM | (≤9 frames) | n/a | **EXCLUDED — partial/leftover/rerun-duplicate** (ADR-029: keep canonical non-l2rerun) |

**Disk-purge reality:** of ~70 CPU-WRF run dirs, only the rows in §1b/§1d retain any `wrfout_d02`
(the Gen2 nightly purges output after ~2 days). All other dirs keep only `wrfinput`/`wrfbdy`/
`namelist.input`. This is why a 70-dir corpus collapses to **3 usable days**.

### 1e. GPU-only / missing

- **GPU-only:** none. Every emitted GPU `wrfout` (`proofs/m20/tost_run/gpu_wrfout/{case1_L2,
  case1_L3,case2_L2,case3_L3}`) is paired to a CPU run; there is no orphan GPU forecast.
- **Missing (the gap):** 12 more independent CPU-WRF days to reach n=15; 24 more to reach n≈27.
  All would be MAM (§4).

---

## 2. Power analysis (honest)

| Quantity | Value |
|---|---|
| ADR-029 floor / target | **n ≥ 15** (underpowered) / **n ≈ 27** (10% RMSE diff at 20% σ) / n=30 (margin) |
| Current independent n | **3 MAM days** |
| Reported `tost_aggregate.json` n | 4 (scoring-unit lead-block deltas — **NOT** independent days; do not cite as n) |
| Gap to 15 | **+12 days** |
| Gap to 27 | **+24 days** |
| Season coverage | **MAM only (1 of 4 seasons)** → gate is **single-season MAM**, never "seasonal" |

### 2a. Empirical σ vs the 20% planning value (the key new datum)

From the frozen n=4 run (`tost_aggregate.json`), empirical σ of the paired ΔRMSE and the
ADR-029 provisional (20%) planning σ, with required-n recomputed (paired t, α=0.05, β=0.20,
`MDE(n) = (t₀.₉₅,ₙ₋₁ + t₀.₈₀,ₙ₋₁)·σ/√n`):

| Var | empirical σ | planning σ (20%) | emp/plan | mean ΔRMSE | frozen margin | req n (empirical σ) | req n (ADR planning) |
|---|---|---|---|---|---|---|---|
| **T2** | 0.4598 K | 0.4297 K | **1.07× (ABOVE)** | **+1.0546 K** | 0.2149 K | **30** | 27 |
| U10 | 0.0802 m/s | 0.4613 m/s | 0.17× | +0.1345 m/s | 0.2306 m/s | 3 | 27 |
| V10 | 0.1103 m/s | 0.5505 m/s | 0.20× | +0.0454 m/s | 0.2752 m/s | 3 | 27 |

### 2b. Honest consequences (predeclared, binding)

1. **T2 is the binding blocker, and it is a BIAS problem, not a power problem.** The T2 mean
   paired ΔRMSE (+1.05 K) is **already 4.9× the frozen margin (0.215 K)**. TOST can only pass
   when the mean delta sits *inside* the margin with adequate power; here the mean is far
   *outside* it. **No backfill fixes this** — growing n only tightens the CI around a delta
   that is itself out of bounds. T2 equivalence requires first closing the GPU warm bias
   (V0.2.0 Wave 1 **P1-4a** MYNN/HFX debt + the v0.1.0 d03 HFX/over-flux story). Empirical
   T2 σ is also **above** the planning value (1.07×), so even on bias the required n rises to
   ~30, confirming ADR-029's "report UNDERPOWERED if empirical σ exceeds planning" trigger.
2. **U10/V10 small empirical σ is NOT a license to shrink n.** The tiny σ (0.08–0.11) comes
   from only **3 correlated MAM days** (and 4 lead-blocks treated as units). With independent
   n=3 the variance is barely estimable and is optimistically low; the ADR-029 planning n≈27
   stands as the design target until σ is re-estimated from **≥15 independent days**. Reporting
   "U10/V10 need only n=3" would be a gameable artifact of the small correlated sample —
   explicitly rejected here.
3. **Single-season rule (ADR-029, binding):** the corpus is 100% MAM. Any result is labeled
   **"single-season MAM (spring trade-wind) equivalence check, n=NN"** and **NEVER "seasonal."**
   The seasonal headline is gated on multi-season coverage + stats-reviewer sign-off (Option C).
4. **Release language (binding):** until n ≥ 15 independent MAM days AND T2 bias is closed, the
   release states **"TOST UNDERPOWERED (n=3, single-season MAM) AND non-equivalent on T2."**
   The current n=4 aggregate already records `all_variables_equivalent=false`,
   `verdict=NOT_EQUIVALENT_OR_UNDERPOWERED` — that honest verdict is preserved, not amended away.

---

## 3. Frozen ADR-029 margins (verbatim — DO NOT re-derive)

These are the **M8.A predeclared margins**, frozen by ADR-029. They were set BEFORE any case
was scored (margin = 10% of the local CPU-WRF-v4-vs-AEMET RMSE benchmark). **Any loosening
requires a follow-up ADR before M21; they must never be re-derived to fit a result.**

Quoted verbatim from `.agent/decisions/ADR-029-STATISTICS-DESIGN-TOST.md` (Decision table):

| Metric | CPU WRF RMSE benchmark | Equivalence margin | Margin source |
|---|---:|---:|---|
| T2 RMSE | 2.148692978020805 K | +/-0.2148692978020805 K | post_iter2_skill_diff.json:31 |
| U10 RMSE | 2.3064713972582305 m/s | +/-0.23064713972582307 m/s | post_iter2_skill_diff.json:62 |
| V10 RMSE | 2.7523205379208537 m/s | +/-0.2752320537920854 m/s | post_iter2_skill_diff.json:93 |

Power-design constants (also frozen, ADR-029 Power Analysis):

| Metric | Provisional σ (20%) | MDE at n=15 | MDE at n=30 | Required n for 10% RMSE diff |
|---|---:|---:|---:|---:|
| T2 RMSE | 0.429738595604161 K | 0.29174914682029846 K | 0.20033130121985668 K | 27 |
| U10 RMSE | 0.46129427945164614 m/s | 0.3131722722598272 m/s | 0.21504161877269762 m/s | 27 |
| V10 RMSE | 0.5504641075841707 m/s | 0.3737095885397448 m/s | 0.25660993002532245 m/s | 27 |

- Test: **paired TOST** on case-level RMSE deltas (RMSE_GPU − RMSE_CPU); equivalence accepted
  **only if BOTH one-sided tests reject at α=0.05 for EVERY required variable**. α=0.05, β=0.20.
- **Q2 / PBLH are NOT in ADR-029's frozen TOST margin set.** ADR-029 freezes T2, U10, V10 only.
  Q2/PBLH/LH/QFX appear in V0.2.0 Wave-1 as **no-regression / oracle-parity** checks (P1-4a),
  not as TOST-margin variables. If a future ADR adds Q2/PBLH to the TOST set it must predeclare
  their margins the same way (10% of CPU-WRF benchmark) **before** scoring. **Until then the
  frozen TOST variable set is exactly {T2, U10, V10}.**

---

## 4. Backfill plan (PLAN ONLY — nothing launched here)

### 4.0 What is computable now vs needs new CPU-WRF

- **Computable from EXISTING GPU + CPU pairs (no new run):** the n=3 / 4-unit TOST is already
  computed and frozen (`tost_aggregate.json`). The 2026-05-30 L3 (§1d) is **CPU-only on disk**
  — it needs a **GPU forecast** (one sequenced GPU slot, ~10 min L3) to become a paired 4th day;
  no new *CPU-WRF* run. **Capture it before the nightly purge or it is lost (§4.1).**
- **Needs NEW CPU-WRF (the long-pole, 28 cores → PRINCIPAL DECISION):** every day beyond the
  3 (4) on disk requires re-running preserved AIFS forcing through WPS→real→wrf. **This consumes
  the 28 CPU cores reserved for the nightly run — flag as principal-decision; not launched here.**

### 4.1 Stop-the-bleed (zero-compute, do FIRST)

The nightly purges `wrfout` after ~2 days, which is why 70 dirs → 3 usable days. **Before any
backfill**, the Gen2 nightly must **retain (or archive thinned to T2/U10/V10/RAINNC/RAINC +
XLAT/XLONG) the wrfout of any enrolled validation case** into a stable corpus dir, else the
backfill re-depletes and the TOST is irreproducible. (Owner: Gen2 pipeline; not model code;
principal-gated since it touches the nightly.) **Immediate casualty if skipped:** the 2026-05-30
L3 candidate (§1d) disappears within ~2 days.

### 4.2 Preserved AIFS forcing available for backfill (no network/fetch)

`/mnt/data/canairy_meteo/runs/forcing_cases/` retains raw AIFS GRIB2 forcing for **41 init
dates** (2 in March, the rest a near-continuous 2026-04-28 → 2026-05-30 daily MAM series).
The expensive externally-dependent AIFS fetch is already done; backfill needs only local
WPS(metgrid) → real.exe → wrf.exe. **~20 forcing dates have init ≥ 2026-05-08 that overlap the
hourly obs window** → enough to reach n≈15–20 MAM, but **MAM only** (no JJA/SON/DJF forcing
exists — those cannot be backfilled, only grown going-forward).

### 4.3 Sequenced backfill to move n (L2 72h is the right unit — one run yields all 3 lead blocks)

Measured CPU-WRF cost (28-rank, from rsl logs / wrfout mtimes on this box): **L2 72h ≈ 5.2 h**
wall (cheaper than L3 24h ≈ 17.7 h; L2's 9/3-km d02 is the scoring domain for both). GPU pairing
cost is small (~22 min/L2 on the RTX 5090), one sequenced GPU slot — **not** the binding cost.

| Step | target n (MAM days) | new CPU-WRF L2 72h runs | CPU wall (serial, 5.2 h/run) | disk (~1.8 GB/run) | gate |
|---|---|---|---|---|---|
| S0 | enroll 05-30 (existing CPU) | +0 CPU runs (1 GPU pair) | 0 | 0 | **GPU slot only — capture before purge** |
| S1 | 15 (floor) | ~11 obs-overlap May dates | ~57 CPU-h ≈ 3 background nights | ~20 GB | **principal-decision (28 cores)** |
| S2 | 20 | +5 dates | ~26 CPU-h ≈ 1–2 nights | ~9 GB | principal-decision |
| S3 | ~21 (all preserved obs-overlap May dates) | +1 | ~5 CPU-h | ~2 GB | **n≈27 NOT reachable from May forcing alone** |

**Suggested S1 date sequence** (preserved-forcing dates with init ≥ 2026-05-11, fully
obs-covered, excluding the 3 days already in corpus and any l2rerun duplicate): 05-11, 05-12,
05-13, 05-14, 05-15, 05-16, 05-17, 05-18, 05-19, 05-20, 05-22 → 11 runs → corpus = 3 + 11 = 14;
add 05-23 → 15 (floor). Then 05-24/05-25/05-26/05-27 → 19–20 for S2. (05-09 already in; 05-28
forcing exists but its wrfout was purged so it re-runs from forcing like the rest.)

**Disk budget:** `/mnt/data` has **217 GB free (92% used)**. A 15-case L2 corpus ≈ 27 GB,
a 20-case ≈ 36 GB — feasible but tight; **thin scored wrfout to the surface fields + grid** and
archive promptly (§4.1). Do **not** retain full 3-D L2 output for 20 cases (~36 GB raw → fine,
but full-field would balloon).

### 4.4 The only route to a SEASONAL claim (Option C — going-forward, not backfillable)

Enroll one fixed L2 72h CPU-WRF + AIFS case per night into the nightly with wrfout retained
(thinned). Accrues ~30 cases/season at zero human cost; after ~12 months → genuine
≥30/season multi-season corpus. **This is the ONLY path to a true "seasonal" TOST headline and
must start now to have JJA/SON/DJF by a future seasonal milestone.** (Principal-gated.)

### 4.5 Backfill does not substitute for the T2-bias fix

Per §2b(1): even a perfect n=27 MAM corpus will **fail T2 equivalence** while the GPU runs
~+1 K warm. **Sequence: (a) close the warm-T2 / MYNN-HFX bias (P1-4a) → (b) re-emit GPU pairs
→ (c) backfill n → (d) score TOST.** Backfilling before (a) burns 28-core compute on a corpus
that cannot pass T2. Recommend: do S0 (free) + stop-the-bleed (§4.1) now; defer S1+ CPU
backfill until after the T2-bias fix lands, then size n from the *post-fix* empirical σ.

---

## 5. Exclusion log spec (predeclared — makes the final TOST non-gameable)

A case/unit/block is **excluded** only by a rule fixed BEFORE GPU-vs-CPU deltas are inspected,
and each exclusion is recorded with reason + reviewer signature. **Exclusions decided after
seeing the delta are forbidden** (that is the gaming vector ADR-029 closes).

**Predeclared exclusion rules (from ADR-029 + this manifest):**

1. **Low-N block:** exclude `(case, variable, lead-block)` if complete-pair count **< 30 rows**.
   (Status field `EXCLUDED_LOW_N`; already enforced by the scorer — e.g. case2_L2 0-24h pre-obs,
   all 24-48h/48-72h of the 24-h L3 cases.)
2. **Intra-case grid mismatch:** exclude a case if its GPU score grid shape **differs from its
   own CPU run's d02 grid**. (All current cases pass — GPU is initialized on the CPU run's grid.
   Cross-*case* grid heterogeneity is logged in §1a, NOT excluded.)
3. **No obs-covered lead:** exclude lead hours with no overlapping hourly AEMET obs
   (obs begin 2026-05-11 09z). → excludes the entire 2026-05-09 L3 (ends 05-10 18z) and the
   05-09 L2 0-24h block.
4. **Rerun duplicate:** when a canonical and an `l2rerun`/re-issued dir both exist for a date,
   keep the **canonical (non-l2rerun)** run; exclude the duplicate.
5. **Incomplete history:** exclude any run with an incomplete `wrfout_d02` series (fewer frames
   than its fhours+1; e.g. the 2,3,9,20-frame leftover dirs in §1d).
6. **Sibling de-duplication for the power n:** L2 and L3 of the SAME init day count as **one
   independent sample** for the ADR-029 n (same synoptic situation). Both may be scored as
   units, but n_independent = distinct init days. (This is why n=3, not 4/5.)
7. **Unrepresentative-after-deletion:** if complete-pair deletion leaves a case/variable
   unrepresentative, mark it excluded **before** looking at deltas; **stats-reviewer signs it**
   (ADR-029 missing-data rule). No imputation of obs/CPU/GPU/station — complete-pair deletion only.

**Reviewer requirement (ADR-029, binding):** the M20 corpus build and the M21 TOST close each
require a **statistics reviewer (Opus or Gemini agy)** to sign the exclusion log. No TOST
verdict closes without that signature.

**Non-gameability invariants:** (i) margins frozen in §3 pre-scoring; (ii) exclusion rules
frozen here pre-delta; (iii) independent-n = distinct init days (no unit inflation);
(iv) single-season label mandatory; (v) the honest `NOT_EQUIVALENT_OR_UNDERPOWERED` verdict is
preserved, never amended; (vi) backfill compute is principal-gated, not self-launched.

---

## 6. Artifacts referenced (read-only; none modified)

- Frozen margins/power: `.agent/decisions/ADR-029-STATISTICS-DESIGN-TOST.md`, `proofs/m20/tost_design.json`.
- Frozen TOST result (n=4 units / 3 days): `proofs/m20/tost_run/tost_aggregate.json`,
  `tost_campaign_result.json`, `paired_score_{case1_L2,case1_L3,case2_L2,case3_L3}.json`.
- Case manifest (3 days): `proofs/v010_validation/v010_cases_manifest.json`.
- CPU baseline manifest (grid/config/sha): `proofs/m20/cpu_baseline_manifest.json`.
- Prior gap docs (corroborated): `proofs/m20/tost_campaign_plan.md`, `proofs/m20/seasonal_gap_assessment.md`.
- Disk truth (2026-06-01): CPU-WRF dirs under `/mnt/data/canairy_meteo/runs/{wrf_l2,wrf_l3}`;
  preserved forcing `/mnt/data/canairy_meteo/runs/forcing_cases/` (41 dates); `/mnt/data` 217 GB free.
- Spec: `.agent/decisions/V0.2.0-PLAN.md` (Wave −1 + 3-case real GPU-vs-CPU compare).
