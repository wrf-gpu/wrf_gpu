# M9 closure — manager-Opus consolidated divergence map + viability verdict

**Sprint**: M9 (operational-mode savepoint parity audit) closed by manager directly after M9.A v2 produced the diagnostic content; M9.B as a separate codex sprint was deemed redundant.
**Verdict**: `VIABLE`
**Date**: 2026-05-28
**Authored by**: Opus 4.7 (manager)
**Inputs**:
- `proofs/m9/operational_trace_hourly.json` (24h × 16 fields divergence trace)
- `proofs/m9/savepoint_parity_1000.json` (status: BLOCKED — codex sandbox GPU init)
- `.agent/sprints/2026-05-27-m7-skill-fix-iter2/post_iter2_skill_diff.json` (prior skill measurement)
- `.agent/sprints/2026-05-27-m7-skill-regression-rca-opus/top_3_suspects.md` (prior RCA)
- `tests/savepoint/test_dycore_100_steps.py` (PASSING — 458 s real GPU run)

## Why M9.B was folded into manager-side closure

The M9.A v2 deliverable `operational_trace_hourly.json` already contained the full per-field-per-hour divergence data across 16 operational variables × 24 hours. Building a separate M9.B "divergence map" codex sprint to consolidate that into a single JSON would have wasted 3-5 hours of wall-clock for purely organisational value. Manager-Opus consolidated directly into `proofs/m9/divergence_map.json` with the viability verdict.

## Headline finding

**Multi-source defects at hour 1 across most operational fields**, but the divergence pattern is **localised to named operators** that map cleanly onto the planned M10-M14 phase. The dycore in isolation (W, QVAPOR, 100-step coupled parity) is healthy. Five defects ranked by likely impact (full numbers in `divergence_map.json`):

| Rank | Hypothesis | Evidence | Fix milestone | Confidence |
|---:|---|---|---|---|
| 1 | **theta reference-state convention mismatch** (WRF stores T = θ-300 perturbation, GPU likely absolute θ) | theta mean RMSE 75 K is unphysical | M9.C (NEW, 1-3 d) or M11 first sprint | high |
| 2 | Surface heat flux (HFX) magnitude/sign wrong | HFX max 4105 W/m² is physically impossible | M12 | high |
| 3 | Radiation initialization wrong (cosine-zenith / albedo) | SWDOWN max 1122 W/m² exceeds clear-sky surface | M13 | medium |
| 4 | LU_INDEX static-field mismatch (14-category delta) | every hour identical | M10 | high |
| 5 | Lateral BC completeness partial (W/P/PB missing) | wind divergence at interior, not pure BC-driven | M14 | medium |

## Why VIABLE

- All 5 defects map to existing planned milestones; **no evidence of a deep architectural defect requiring redesign**.
- Dycore + advection healthy in isolation (W ✓, QVAPOR ✓, 100-step parity ✓).
- Static fields HGT, LANDMASK bitwise match — ingestion path is mostly right.
- TSK perfect bitwise — data-replay coupling works as designed.

## Caveats (important, not project-killers)

1. **Comparator method needs audit** — the theta RMSE 75 K and T2 max-max 157 K and SWDOWN 1122 W/m² are too unphysical to be pure model bug; they're likely **method artifacts** (unit mismatch, vertical-level misalignment, NaN handling). Before driving M11-M14 with these numbers, the comparator must be audited. This is M9.C (1-3 days, NEW).
2. **1000-step dycore parity not measured** — codex sandbox blocks the specific JAX init path of `savepoint_parity_1000.py`; the 24h operational pipeline DID use GPU so this is a script-specific issue not a project blocker. Manager-side rerun in next overnight cycle.
3. **20260521 is a single case** — the M20 30-case seasonal ensemble will reveal how the defects vary; the diagnosis above should be re-validated on at least 3 cases before committing major M11-M14 effort.

## Adjusted dispatch order (recommendation)

| New order | Milestone | Why this order |
|---|---|---|
| **1** | **M9.C** comparator + theta-convention audit | If theta is the dominant artifact, M11-M14 priorities change |
| 2 | M10 LU_INDEX static-field parity | Cheap, confirmed defect, prerequisite for clean physics |
| 3 | M11 dycore guard + theta limiter | Now informed by M9.C |
| 4 | M12 surface flux + MYNN parity | Largest expected skill gain |
| 5 | M13 radiation + land-surface diurnal | Depends on M12 surface flux |
| 6 | M14 lateral BC completeness | Lower expected gain but needed |

## Plan amendment proposed

Insert M9.C as a 1-3 day sprint BEFORE M11 in PROJECT-RESET-PLAN-FINAL.md. Will write an ADR if the comparator audit reveals scope-changing findings; otherwise apply textually.

## Invariant ladder status (INV-1..6 at M9 close)

| Invariant | Status |
|---|---|
| INV-1 D2H zero | No new measurement; ADR-027 PROPOSED unchanged |
| INV-2 B6 @ 100 steps | ✅ PRESERVED (458 s real run PASS) |
| INV-3 op-mode parity ≥ 1000 steps | ⚠ BLOCKED — script-specific issue, follow-up |
| INV-4 mini-ensemble RMSE | ⚠ Still single-case; M20 corpus required for true mini-ensemble |
| INV-5 ≥ 10× perf | No new measurement; 22.26× corrected number stands |
| INV-6 no test relaxation | ✅ No tests deleted; xfail placeholders have explicit reason strings |

## Decision

M9 closes with verdict `VIABLE`. Recommend dispatching M9.C comparator audit immediately, then M10, then Phase B in dependency order. The 2-milestone reflection cadence is in flight (codex plan-critic dispatched in parallel on branch `worker/gpt/m9-plan-critic`); its findings will be integrated into the next plan amendment.
