# MILESTONE M7 CLOSEOUT — AMENDMENT (2026-05-27 post-validation)

**Status: M7-OPERATIONALLY-CLOSED-DOWNGRADED → M7-PENDING-SKILL-FIX**

This amendment **supersedes the publication-readiness claims** in `MILESTONE-M7-CLOSEOUT.md` (commit `1820548`). Two follow-up sprints (honest speedup + L2 d02 replay) found that:

1. The headline **156.82× speedup was inflated** by a timing-record path issue.
2. The GPU forecast has a **material skill regression vs CPU WRF** (+243-440% RMSE on T2/U10/V10), discovered by side-by-side AEMET station scoring.

This amendment documents the corrected state. The original closeout is preserved as a record of what was believed at commit-time; this document overrides any publication-facing claims.

## Corrected headline numbers

| Metric | Original claim | Corrected | Status |
|---|---|---|---|
| GPU 1h warm wall (3km) | 5.71 s | 5.71 s | unchanged ✅ |
| GPU 24h pipeline wall | 324.78 s | 324.78 s | unchanged ✅ |
| D2H inside loop | 0 / 0 bytes | 0 / 0 bytes | unchanged ✅ |
| Restart bitwise | PASS | PASS | unchanged ✅ |
| Repeatability bitwise | PASS | PASS | unchanged ✅ |
| 1km VRAM headroom | 78% | 78% | unchanged ✅ |
| **Speedup vs CPU baseline** | **156.82×** | **50.20× apples-to-apples** | **CORRECTED** |
| **AEMET T2 RMSE** | (not measured GPU-vs-CPU) | **GPU 7.86 K, CPU 2.15 K (+266%)** | **NEW — REGRESSION** |
| **AEMET U10 RMSE** | (not measured GPU-vs-CPU) | **GPU 11.31 m/s, CPU 2.31 m/s (+390%)** | **NEW — REGRESSION** |
| **AEMET V10 RMSE** | (not measured GPU-vs-CPU) | **GPU 9.44 m/s, CPU 2.75 m/s (+243%)** | **NEW — REGRESSION** |

## Source of the 156× → 50× correction

`.agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/verdict.md`: the original 156.82× came from a timing denominator that double-counted mirrored WRF `Timing for main` records on the wrong source run. The complete 20260521 Gen2 reference (`20260521_18z_l3_24h_20260522T133443Z`) has zero `Timing for main` records in `namelist.output`; the corrected number uses de-duplicated sibling `rsl.error.0000`/`rsl.out.0000` records:

- GPU d02 24h: 324.78 s
- CPU d02-only cumulative WRF timing: 16,305 s
- **Honest ratio: 50.20×**

Other framings (with caveats):
- CPU full 5-nest aggregate: 138.24× (not apples-to-apples)
- CPU d01+d02 minimum physical subset: 102.62× (still includes CPU d01)
- CPU d01-only context: 52.42×

## Source of the skill regression finding

`.agent/sprints/2026-05-27-m7-honest-speedup-skill-diff/gpu_vs_cpu_skill_diff.json`: both GPU and CPU 20260521 d02 wrfouts run through the SAME `gpuwrf.validation.forecast_vs_obs` scaffold, 73 AEMET stations × 24 hours = 1747 joined rows. The scoring code is identical for both inputs; only the wrfout source differs. GPU is 240-440% worse on every metric of every variable.

Independent confirmation: `.agent/sprints/2026-05-27-m7-l2-d02-replay-validation/tier4_rmse_l2_d02.json` ran the GPU port on L2 (the same 3km grid but with L2-d01 boundary forcing) and produced `L2_D02_BOUNDED_FAIL`:
- T2 RMSE 4.07 K (threshold 3 K) — FAIL
- U10 RMSE 10.78 m/s (threshold 7.5) — FAIL
- V10 RMSE 7.83 m/s (threshold 7.5) — FAIL

Same dycore, different boundary source, same kind of failure. This is a dycore/physics issue, not an artifact of the L3 boundary path specifically.

## What the original validation missed

The M7 close declared 6/8 gates done and PIPELINE_GREEN based on:
- Bounds preserved (✅ truly preserved)
- Wall-clock measurement (✅ accurate, but speedup denominator was wrong)
- Restart + repeatability (✅ truly preserved)
- AEMET station scoring was performed on the **GPU output only** and reported as "1747 rows of finite numbers, BIAS/RMSE/MAE measurements" with the worker explicitly noting it was *NOT* a skill claim.

**The error in the original closeout was treating "finite station scores" as evidence of operational skill.** The pipeline integration sprint correctly flagged this caveat ("scoring values are measurements against AEMET, not a skill claim against CPU WRF"); the closeout author (me, Claude Opus 4.7 manager) elevated it to a milestone-close gate anyway. That was wrong. The correct gate is a **side-by-side GPU vs CPU skill comparison**, which is what this amendment's honest-speedup sprint produced.

## What's still real

These claims survive the amendment unchanged:
- The 24h pipeline runs end-to-end without crashing
- D2H inter-kernel = 0 (constitutional invariant ADR-027 holds)
- Restart bitwise PASS
- Repeatability bitwise PASS
- 1km Canary grid fits VRAM (78% headroom)
- Wall-clock measurement methodology (the **GPU** number) is reproducible to CV 0.42%
- Speedup vs CPU baseline of **~50×** (apples-to-apples)

These are real engineering achievements. They are not enough for an operational claim or a publication claim.

## What's NOT publication-ready

- The GPU forecast is **fast but materially less skillful than the CPU baseline** on T2/U10/V10. Publishing the 50× number without disclosing the +243-440% skill regression would be dishonest.
- The 156× number must not be cited anywhere; even the M7-CLOSEOUT.md document overstates this.
- "M7 operationally closed" is overclaiming; the correct state is "**M7 perf-infrastructure complete; skill regression discovered; root cause investigation underway**."

## Sprints in flight as of this amendment

- `2026-05-27-m7-skill-regression-rca-opus` (opus tester, architecture angle): boundary-forcing path audit, physics coupling order, surface/SST, radiation cadence, wind damping. ETA 1-3h.
- `2026-05-27-m7-skill-regression-rca-codex` (codex worker, empirical bisection): hour-by-hour deviation, spatial maps, first-hour diff, boundary vs interior, physics on/off bracket. ETA 2-4h.

These will name the most likely root cause. A fix sprint follows.

## What is held / not dispatched

- **L2.1 d01 ingest sprint** (per nest scout's `BACKFILL_NEEDS_NEW_CODE` recommendation) is **HELD** until skill regression is understood. No point dispatching a 9km parent run if the 3km nest has a fundamental forecast-quality issue.
- **Publication / arXiv writing** is **HELD** until skill is recovered or the regression is honestly characterized as a known limitation.
- **Backfill execution** is **HELD** until the skill issue is at least understood (we may still proceed with documented caveat, but that's the user's call).

## Recommended forward path

1. **Wait for RCA reports** (1-4h horizon).
2. **Synthesize** the opus architectural priors vs the codex empirical bisection.
3. **Dispatch fix sprint** scoped to the named root cause.
4. **Re-run honest speedup + skill diff** post-fix.
5. **If skill recovers to within ±20% of CPU**: amend the closeout to M7-PUBLISHED-READY; dispatch L2.1 d01 ingest; proceed with backfills.
6. **If skill does not recover**: characterize the limitation honestly in the publication; ship as "fast prototype dycore with documented skill gap" rather than "operational replacement."

## Honest framing for any external communication right now

If the principal needs to communicate the state externally (e.g. status update to stakeholders), the honest framing is:

> "We have a GPU-native Python port of WRF that runs a 24h Canary 3km forecast in 5.4 minutes wall-clock on a single RTX 5090 — approximately 50× the throughput of our 28-rank CPU WRF baseline on the same workstation. The dycore is numerically stable, restart-continuity is bitwise-exact, and the 1km Canary domain fits with 78% VRAM headroom. **Forecast skill on AEMET station observations is currently materially worse than the CPU baseline; root cause investigation is underway.** Publication is pending skill resolution."

That is the honest version. It is still a substantial result.

## Sign-off

This amendment is committed before any publication or external claim. The original M7-CLOSEOUT.md remains as the record of what was thought correct at the time; this document supersedes it for any forward-facing claim.

Manager (Claude Opus 4.7, 1M context, autonomous overnight loop) — 2026-05-27.
