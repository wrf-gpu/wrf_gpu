# ADR-029 - Statistics Design for TOST Equivalence

**Status:** PROPOSED
**Date:** 2026-05-28
**Decision owner:** Manager
**Applies to:** M20 validation-corpus build and M21 statistical-equivalence close.

## Context

The reset binding goal requires Canary L2/L3 24-72 h RMSE on T2, U10, and V10 to be statistically equivalent to CPU WRF v4 under TOST at predeclared margins on a >=15-case seasonal ensemble, with the speed floor preserved (sprint-contract.md:13). The contract also requires the M20/M21 power design to report n=15 and n=30 behavior and the sample size needed to detect a 10% RMSE difference at alpha=0.05 and beta=0.20 (sprint-contract.md:106).

Current local AEMET side-by-side benchmarks are the only accepted operational benchmark for this ADR. The latest iter2 CPU WRF RMSE values are T2 2.148692978020805 K (post_iter2_skill_diff.json:31), U10 2.3064713972582305 m/s (post_iter2_skill_diff.json:62), and V10 2.7523205379208537 m/s (post_iter2_skill_diff.json:93).

## Decision

Use paired TOST on case-level RMSE deltas, GPU minus CPU, for each metric. Equivalence is accepted only if both one-sided tests reject at alpha=0.05 for every required variable, matching the reset plan's TOST p<0.05 requirement (PROJECT-RESET-PLAN-FINAL.md:87).

Predeclared equivalence margins are set to 10% of the current local CPU WRF RMSE benchmark per variable. These margins are intentionally stricter than the earlier +/-20% single-case skill-recovery gate used for M19 (PROJECT-RESET-PLAN-FINAL.md:80).

| Metric | CPU WRF RMSE benchmark | Equivalence margin | Margin source |
|---|---:|---:|---|
| T2 RMSE | 2.148692978020805 K | +/-0.2148692978020805 K | post_iter2_skill_diff.json:31 |
| U10 RMSE | 2.3064713972582305 m/s | +/-0.23064713972582307 m/s | post_iter2_skill_diff.json:62 |
| V10 RMSE | 2.7523205379208537 m/s | +/-0.2752320537920854 m/s | post_iter2_skill_diff.json:93 |

These are M8.A predeclared margins. M20 may tighten them before cases are scored. Any loosening after this ADR requires a follow-up ADR before M21.

## Paired Design

Pairing key: `case_id x domain x valid_time_utc x lead_hour x station_id x variable`. The scorer must form CPU/GPU/observation complete pairs before aggregation; no side may use a row that the other side lacks. The prior M7 overclaim came from treating finite station scores as skill evidence, so same-scorer, same-mask side-by-side comparison is mandatory (MILESTONE-M7-CLOSEOUT-AMENDMENT.md:56).

Aggregation unit: for each case, domain, lead-hour block, and variable, compute RMSE for CPU and GPU on the exact same complete-pair row set, then store the paired delta `RMSE_GPU - RMSE_CPU`. TOST operates on those case-level paired deltas, not on unpaired station rows.

Missing data: use complete-pair deletion only. Do not impute observations, CPU forecasts, GPU forecasts, or station metadata. If a case/variable loses enough rows that its RMSE is not representative, M20 must mark the case/variable excluded before looking at GPU-vs-CPU deltas and the statistics reviewer must sign the exclusion.

Season stratification: M20 case manifest must label each case by meteorological season and domain family. M21 reports both the pooled TOST result and season-stratified descriptive deltas; if one season dominates the corpus, M21 cannot claim seasonal equivalence without reviewer approval.

## Power Analysis

Let `sigma_v` be the empirical standard deviation of case-level paired RMSE deltas for variable `v`. Until M20 estimates `sigma_v`, planning uses a conservative provisional sigma equal to 20% of the CPU WRF RMSE benchmark. The 20% scale is taken from the local skill-gate tolerance fraction (post_iter2_skill_diff.json:6) and the M19 single-case recovery gate (PROJECT-RESET-PLAN-FINAL.md:80).

Formula for planning MDE:

`MDE(n) = (t_0.95,df=n-1 + t_0.80,df=n-1) * sigma_v / sqrt(n)`.

With alpha=0.05 and beta=0.20 from the sprint contract (sprint-contract.md:106), the planning coefficients are 0.6788991023953389 x sigma_v at n=15 and 0.46617013986890066 x sigma_v at n=30.

| Metric | Provisional sigma | MDE at n=15 | MDE at n=30 | Required n for 10% RMSE difference | Sources |
|---|---:|---:|---:|---:|---|
| T2 RMSE | 0.429738595604161 K | 0.29174914682029846 K | 0.20033130121985668 K | 27 | post_iter2_skill_diff.json:31; post_iter2_skill_diff.json:6; sprint-contract.md:106 |
| U10 RMSE | 0.46129427945164614 m/s | 0.3131722722598272 m/s | 0.21504161877269762 m/s | 27 | post_iter2_skill_diff.json:62; post_iter2_skill_diff.json:6; sprint-contract.md:106 |
| V10 RMSE | 0.5504641075841707 m/s | 0.3737095885397448 m/s | 0.25660993002532245 m/s | 27 | post_iter2_skill_diff.json:93; post_iter2_skill_diff.json:6; sprint-contract.md:106 |

Interpretation: with the conservative 20% sigma assumption, n=15 is underpowered for a 10% RMSE difference, while n=30 is close to the target. M20 must compute empirical `sigma_v` before M21. If empirical sigma is materially above the 20% planning value, M21 either expands the corpus or reports the TOST result as underpowered.

## Reviewer Requirement

M20 and M21 both require a statistics reviewer, either Opus or Gemini agy. The reset plan makes statistics review mandatory for M20/M21 (PROJECT-RESET-PLAN-FINAL.md:146), and the M8.A sprint contract repeats that requirement (sprint-contract.md:107).

## Consequences

- A non-significant paired difference is not equivalence.
- TOST margins are fixed before M20 scoring.
- Case list, season labels, complete-pair masks, effect sizes, confidence intervals, and p-values are part of the proof object.
- M21 cannot close on a pooled result alone if missingness or seasonal imbalance changes the interpretation.
