# TOST Readiness — honest seasonal-equivalence accounting

The binding statistical-equivalence goal (ADR-029 / reset plan) is a **paired TOST**
showing Canary L2/L3 24–72 h RMSE on T2/U10/V10 statistically equivalent to CPU-WRF
v4 at predeclared margins on a **≥15-case (target ≥30) seasonal ensemble**. This
table states honestly **why v0.1.0 does NOT yet make a seasonal-equivalence claim.**

Sources: `proofs/m20/tost_design.json`, `proofs/m20/case_manifest.json`,
`proofs/m20/seasonal_gap_assessment.md`.

## Corpus inventory (the shortfall)

| Quantity | Value |
|---|---:|
| CPU-WRF run directories on disk | 66 (30 L2 + 36 L3) |
| Runs that still retain full wrfout output | **5** (2 L2 @72 h, 3 L3 @24 h) |
| Runs scoreable against hourly obs *now* | **4** |
| Season represented | **MAM (spring) only — 100 % of all 66 run dirs** |
| Usable init dates | 2026-05-09, 05-21, 05-28 (3 distinct days) |
| Hourly obs window | 2026-05-11 09z → 2026-05-29 15z |
| AEMET stations (total / hourly-reporting) | 106 / 73 |

**Two independent blockers:**
1. **Output purge** — the Gen2 nightly pipeline keeps wrfout ~2 days; 61 of 66 run
   dirs have had wrfout deleted (only wrfinput/wrfbdy/namelist survive). Available
   CPU baseline = 5 cases, far short of ≥15/≥30.
2. **Seasonal narrowness** — every case is mid-to-late May 2026 (MAM). Even a
   fully-backfilled May corpus proves *spring trade-wind* equivalence, not
   *seasonal* equivalence.

## Predeclared TOST design (frozen, prevents post-hoc margin-shopping)

| Parameter | Value | Source |
|---|---|---|
| Test | paired TOST on case-level RMSE deltas (RMSE_GPU − RMSE_CPU) per variable | `tost_design.json` |
| Accept rule | BOTH one-sided tests reject at α=0.05 for EVERY required variable | " |
| α / β | 0.05 / 0.20 | " |
| Equivalence margin | 10 % of CPU-WRF RMSE benchmark | " |
| Provisional σ | 20 % of CPU-WRF RMSE (planning value) | " |
| Margin source | CPU-WRF v4 vs AEMET, same scorer (post-iter2 skill diff) | " |

### Predeclared per-variable margins and required n

| Variable | CPU-WRF RMSE benchmark | equivalence margin | MDE @ n=15 | MDE @ n=30 | required n for 10 % MDE |
|---|---:|---:|---:|---:|---:|
| T2 (K) | 2.149 | 0.215 | 0.292 | 0.200 | **27** |
| U10 (m/s) | 2.306 | 0.231 | 0.313 | 0.215 | **27** |
| V10 (m/s) | 2.752 | 0.275 | 0.374 | 0.257 | **27** |

n=15 is underpowered for a 10 % MDE; n≈27 reaches it; n=30 has margin. M20 must
compute **empirical σ** from real paired deltas before M21; if empirical σ
materially exceeds the 20 % planning value, expand the corpus or report TOST as
underpowered.

## Readiness verdict

| Requirement | Required | Available now | Met? |
|---|---|---|:--:|
| Scoreable cases | ≥15 (target ≥30, n≈27 for 10 % MDE) | **4–5** | ❌ |
| Seasonal coverage | ≥2 seasons for a *seasonal* claim | **1 (MAM only)** | ❌ |
| Predeclared margins | frozen before looking at GPU-vs-CPU deltas | **yes (frozen)** | ✅ |
| Stats reviewer sign-off | M20 + M21 require Opus or agy reviewer | not yet executed | ⏳ |
| Empirical σ from real deltas | before M21 | **[PLACEHOLDER: needs M20 paired-delta run]** | ❌ |
| Computed TOST p-values + 90 % CIs | per variable | **[PLACEHOLDER: needs M20/M21 scorer run]** | ❌ |

**Honest statement for the paper:** v0.1.0 demonstrates near-CPU-WRF RMSE and
persistence skill on **3 distinct May days**, but **cannot and does not claim
statistical (TOST) seasonal equivalence.** A pooled TOST on a May-only corpus may
not be reported as seasonal equivalence without explicit reviewer sign-off
(ADR-029 season-stratification rule).

## Recoverable backfill (the good news)

`/mnt/data/canairy_meteo/runs/forcing_cases/` retains raw AIFS GRIB2 forcing for
**39 init dates** (the expensive external-fetch stage is already done/preserved).
**21 of those dates overlap the hourly-obs window** (2026-05-08 → 05-28) and can be
re-run via the local WPS→real.exe→wrf.exe chain (no network), reaching n≥15 and
approaching n≈27/30 — but **still May-only** (spring trade-wind equivalence, not
seasonal). Cross-season backfill needs new forcing fetches. Source:
`proofs/m20/seasonal_gap_assessment.md`.
