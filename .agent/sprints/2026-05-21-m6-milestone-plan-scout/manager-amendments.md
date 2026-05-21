# M6 Milestone Plan — Manager Amendments (Integrating Codex Critic)

**Manager**: Claude Opus 4.7 (1M-context)
**Date**: 2026-05-21 11:42
**Inputs**:
- Scout plan: `worker/codex/m6-milestone-plan-scout @ 3392d04` — `m6-milestone-plan.md` (26 KB)
- Codex critical-review: `.agent/sprints/2026-05-21-m6-milestone-plan-scout/critical-review-codex.md` (RATIFY-WITH-AMENDMENTS, 10 edits)
- User directive: "planning and closing is manager job + 1 gpt agent" (this doc IS the manager integration)

## Verdict: RATIFY-WITH-AMENDMENTS (final manager decision)

Adopt all 10 critic amendments. Sequence and proof objects are now load-bearing.

## Amended sprint sequence

| Phase | Sprint | Status | Wall | Notes |
|---|---|---|---|---|
| Prologue | M5-S1.y Thompson HLO+residuals | ✓ CLOSED (GRAY-ZONE) | 1h | done |
| Prologue | M5-S2.x MYNN follow-ups | ✓ CLOSED (ACCEPT) | 50m | done |
| Prologue | M5-S3.x RRTMG transfer-solver | ✓ CLOSED (GROUNDWORK-PHASE-2) | 35m | done |
| Prologue Phase 2 | M5-S3.y RRTMG setcoef+taumol+Planck-attempt-1 | Opus reviewer in flight; worker self-rejected (budget burst) | 24m | **REJECT-bounded expected** |
| Prologue Phase 2 | **M5-S3.z RRTMG intermediate-oracle sprint** (NEW, per worker §3 recommendation) | STUB queued; dispatch after M6-S1 Opus closes | 16-24h | Extract per-band WRF harness output (`taug, taur, fracs, plank*`) BEFORE more branch expansion. Validate JAX branches against intermediate oracles. |
| M6 IMPL | M6-S1 coupled interface freeze | Opus reviewer in flight | 22m | Worker self-PASS all AC; reviewer to confirm + flag boundary-forcing gap per critic amendment #3 |
| M6 IMPL | **M6-S2a Gen2 backfill accessor + d02 boundary replay** (NEW, per critic amendment #2) | STUB queued; dispatch parallel with M5-S3.z after M6-S1 Opus closes | 12-18h | Without `wrfbdy_d02` extraction, M6-S2 cannot be a real 24h forecast — only a closed-boundary diagnostic |
| M6 IMPL | M6-S2 coupled forecast driver | queued; depends on M6-S1 + M6-S2a | 24-36h | Smoke 1h → 6h → 24h d02 with boundary replay |
| M6 IMPL | M6-S3 surface layer + bounded Noah-MP | queued; depends on M6-S2 smoke | 30-48h | Per critic amendment #4: enumerated Noah-MP subset memo BEFORE code; radiation-conditioning feasibility artifact |
| M6 VAL | M6-S4 Tier-2 coupled invariants | queued; parallel with S5/S6/S7 after S2 smoke | 16-24h | Per critic amendment #8: external/cross-implementation budget oracle (no self-consistency) |
| M6 VAL | M6-S5 ADR-007 4× verdict | queued; parallel | 12-20h | Per critic amendment #9: fair Gen2 CPU denominator (domain-scoped) + concrete FAIL fallback ladder |
| M6 VAL | M6-S6 Tier-3 TSC1.0 | queued; parallel | 18-30h | Per critic amendment #7: controlled dt-refinement reduced case (not `wrf_l2 vs wrf_l3` config-noise) |
| M6 VAL | M6-S7 Tier-4 probtest prototype | queued; parallel | 18-30h | Per critic: prototype label, stratification by land/sea/elevation |
| M6 CLOSE | M6-S8 operational Gen2 comparison + closeout | queued; serial | 24-36h | Per critic amendment #10: CPU-vs-observation as binding gate (not `max(...)`); GREEN/PARTIAL/BLOCKED/FAIL statuses |

## Critical-path read

**Total M6 wall (with prologue + critic amendments)**:
- M5-S3.z (16-24h) + M6-S1 close (small) + M6-S2a (12-18h) + M6-S2 (24-36h) + M6-S3 (30-48h) + validation max-lane (max of S4/S5/S6/S7 = 30h) + S8 (24-36h)
- Critical path: **~150-200h**, vs scout's 144-210h — same order of magnitude
- Calendar with 2 codex workers + Opus review + no major failures: **7-10 days**

## Big-picture sequence to end goal

The Canary 3km/1km daily operational forecast (PROJECT_CONSTITUTION immutable goal) requires:

1. **M6 close GREEN** with: U10/V10/T2 RMSE ≤ CPU-vs-obs noise, ADR-007 4× pass, all Tier-2/3/4 gates clean, observation comparator working.
2. **M7 Canary operational v0**: 3km then 1km pipeline, I/O, restart, post-processing, daily-run cadence, operational verification.
3. **M8 forkable release**: docs, packaging, public review.

Smart large steps to get there:

- **NOW** (waiting on 2 Opus reviewers): manager prep — write M5-S3.z + M6-S2a sprint contracts so they dispatch the moment M6-S1 closes.
- **Next 4-8h** (after M6-S1 Opus accepts): dispatch M5-S3.z (codex) + M6-S2a (codex) in parallel. Two heavy sprints, file-disjoint.
- **Next 8-16h** (after M5-S3.z + M6-S2a close): dispatch M6-S2 forecast driver (codex) + M6-S3 surface (codex) in parallel. Two heavy sprints, file-disjoint.
- **Next 16-30h** (after M6-S2 smoke): dispatch M6-S4 + M6-S5 + M6-S6 + M6-S7 in 4-way parallel validation (file-disjoint per ADR-010).
- **Final**: M6-S8 operational closeout (serial). If GREEN → M7 prep. If PARTIAL → remediation sprint(s) before M7.

## Cross-cutting amendments (critic items 5-10)

- All sprint contracts must include exact JSON proof-object schemas (paths + fields).
- One sprint owns shared validation I/O / regridding / loaders / domain masks (proposed: M6-S2a; cite in its contract).
- File-ownership freeze (M6-S1 ADR-010) is HARD — S3 must not opportunistically edit coupling adapters.

— Manager, 2026-05-21 11:42
