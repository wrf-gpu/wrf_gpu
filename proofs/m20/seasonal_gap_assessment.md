# M20/M21 Seasonal Gap Assessment

**Date:** 2026-05-29
**Owner:** data/validation engineer (Opus 4.8)
**Scope:** What additional CPU-WRF runs are needed for a defensible *seasonal* TOST
equivalence claim, and whether the Gen2 nightly pipeline can backfill them.
**Binding goal (ADR-029 / reset plan):** Canary L2/L3 24–72 h RMSE on T2/U10/V10
statistically equivalent to CPU WRF v4 under paired TOST at predeclared margins on a
**≥15-case (target ≥30) seasonal ensemble**, speed floor preserved.

---

## 1. Current corpus state (honest)

| Quantity | Value |
|---|---|
| CPU-WRF run directories on disk | 66 (30 L2 + 36 L3) |
| Runs that still retain full wrfout output | **5** (2 L2 @72 h, 3 L3 @24 h) |
| Runs scoreable against hourly obs *now* | **4** |
| Season represented | **MAM (spring) only** — 100 % of all 66 run dirs |
| Init dates of usable cases | 2026-05-09, 05-21, 05-28 (3 distinct days) |
| Hourly obs window | 2026-05-11 09z → 2026-05-29 15z |
| AEMET stations (total / hourly-reporting) | 106 / 73 |

**Two independent shortfalls block the M20/M21 claim today:**

1. **Output purge.** The Gen2 nightly pipeline keeps wrfout for only the most recent
   ~2 days; 61 of 66 run dirs have had their wrfout deleted (only `wrfinput`/`wrfbdy`/
   `namelist.input` survive). So the *available* CPU baseline is 5 cases, not 66 — far
   short of the ≥15/≥30 requirement.
2. **Seasonal narrowness.** Every case — surviving or purged — is mid-to-late **May 2026
   (MAM)**. Even a fully-backfilled May corpus proves *spring trade-wind* equivalence, not
   *seasonal* equivalence. The trade-wind regime over the Canaries is itself seasonally
   modulated (stronger/steadier in JJA, more synoptic variability in DJF/SON).

A pooled TOST on a May-only corpus **cannot** be reported as a seasonal equivalence claim
without explicit reviewer sign-off (ADR-029 season-stratification rule).

## 2. What is recoverable cheaply (the good news)

`<DATA_ROOT>/canairy_meteo/runs/forcing_cases/` retains the **raw AIFS GRIB2 forcing for 39
init dates** (2026-03-26, 03-27, and a near-continuous 2026-04-28 → 2026-05-28 daily
series). The expensive, externally-dependent stage — fetching AIFS — is **already done and
preserved**. Backfill therefore only needs the local compute chain WPS(metgrid) → real.exe
→ wrf.exe; no network/data dependency.

- **21 of those forcing dates overlap the hourly-obs window** (2026-05-08 → 2026-05-28),
  i.e. they can be re-run *and* scored against hourly T2/U10/V10. This is enough to reach
  the **n≥15** floor and approach the **n≈27 / n=30** target purely by re-running May cases.
- Output size per full run: **~1.8 GB (L2, 72 h)**, **~1.3 GB (L3, 24 h)**. Disk free on
  `<DATA_ROOT>`: ~261 GB (91 % used). A 30-case L2 corpus ≈ 54 GB — feasible, but tight; the
  re-run pipeline should thin to T2/U10/V10/RAINNC/RAINC + XLAT/XLONG or move scored cases
  to a compressed archive promptly.

## 3. Backfill recommendation

### 3.1 Immediate (reach n≥15, May-only, "spring trade-wind equivalence")
Re-run the **21 obs-overlapping May forcing dates** (L2 72 h is preferred — it yields all
three lead blocks 0-24/24-48/48-72 h from one run, satisfying the 24–72 h goal). On the
28-core CPU WRF (cores 4–31) a 72 h Canary L2 run is well within a nightly slot; ~21 runs ≈
a few nights of background compute. **This unblocks M20 and a *qualified* M21** ("equivalent
in spring trade-wind conditions, n≥15"), which is itself a legitimate viability signal.

**Critical pipeline change:** disable or lengthen the wrfout purge for cases entered into
the validation corpus, OR have the nightly job copy scored wrfout into
`<DATA_ROOT>/wrf_gpu2/corpus/wrfout_archive/<case_id>/` (thinned to the 5 surface fields +
grid) before deletion. Without this, the corpus will re-deplete and M21 will not be
reproducible.

### 3.2 For a *defensible seasonal* claim (the real requirement)
A May-only corpus, however large, does not satisfy "seasonal." To claim seasonal
equivalence the corpus must sample multiple meteorological seasons:

| Season | Recommended cases | Source |
|---|---|---|
| MAM (have) | 15–30 May 2026 (backfill from forcing_cases) | local re-run, no fetch |
| JJA | ≥8 cases across Jun–Aug 2026 | **Gen2 nightly, going forward** |
| SON | ≥8 cases across Sep–Nov 2026 | Gen2 nightly, going forward |
| DJF | ≥8 cases across Dec 2026–Feb 2027 | Gen2 nightly, going forward |

The cleanest path: **enroll a fixed daily (or every-other-day) CPU-WRF + AIFS-forcing case
into the Gen2 nightly pipeline as a permanent "validation case," with wrfout retained.** The
nightly already runs WRF on cores 4–31; adding one retained L2 72 h case per night
accumulates ~30 cases/season with zero extra human effort. After ~12 months this yields a
genuinely seasonal ≥30-case-per-season corpus.

**Interim honest framing for M21:** report the pooled TOST result AND season-stratified
descriptive deltas, and label the conclusion exactly as the evidence supports — most likely
"equivalent under MAM/spring trade-wind conditions, n=NN" until JJA/SON/DJF cases exist.
Per ADR-029, the seasonal-equivalence headline is gated on multi-season coverage + reviewer
approval; it must not be claimed from a single-season corpus.

## 4. Concrete next actions (priority order)

1. **Stop the bleed:** patch the Gen2 nightly to retain (or archive thinned) wrfout for
   enrolled validation cases. (Owner: Gen2 pipeline; not model code.)
2. **Backfill May:** re-run the 21 obs-overlapping forcing dates as L2 72 h on cores 4–31;
   register each in `case_manifest.json` and score with `paired_tost_scorer.py`. → n≥15.
3. **Enroll a permanent nightly validation case** (retained output) to grow JJA/SON/DJF.
4. **M21 reporting discipline:** pooled + season-stratified; seasonal headline only with
   multi-season coverage + stats-reviewer sign-off.

## 5. Bottom line

The *scoring infrastructure* is ready (manifests + paired-TOST scorer validated to
reproduce the ADR-029 CPU benchmark exactly). The *corpus* is not: 5 usable cases, all MAM.
The shortfall is **logistical, not fundamental** — the AIFS forcing for 21 obs-overlapping
May dates is preserved, so n≥15 is a few nights of local re-compute away, and the Gen2
nightly can grow the multi-season corpus from here at zero marginal human cost. Until then,
any M21 conclusion must be scoped to *spring trade-wind* conditions, not "seasonal."
