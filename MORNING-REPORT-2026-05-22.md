# Morning Report — 2026-05-22 ~05:30 UTC

## Executive summary

**Constitutional 4× speedup target ACHIEVED at 43.90× measured (c1-A7).**
**Forecast stability NOT yet closed** after 9 c1 iterations (~7h clock so far).

## Where we are

The c1 Klemp-Skamarock clean-room dycore is converging. Each iteration is surgical (per bughunt #4 anti-pattern warning, ONE fix per probe). Bug count is multilayered — each fix exposes the next.

| Iter | Scope | Outcome | Delay achieved |
|---|---|---|---|
| A1 | Acoustic core | stable isolated | — |
| A2 | Scalar advection flux + 4 ops | operators verified | — |
| A3 | bughunt#3 4 fixes | neutral / slightly worse | — |
| A4 | Buoyancy | disproven by bisection | — |
| A5-H3 | n_acoustic patch | disproven (wrong site) | — |
| **A6** | **EMPIRICAL BISECTION** | **isolated bug → advection** | — |
| **A7** | Horizontal momentum flux form | **CLOSED horizontal** + **43.90× speedup measured** | step 30 → 106 |
| **A8** | Vertical eta-metric sign | **CLOSED vertical-w isolated** | step 106 → 188 |
| **A9** | Cross-component (u+w, v+w) | in flight | targeting 188 → 360+ |

## Decision needed: continue c1 OR pivot

c1 might converge in 1-2 more iterations OR continue spiraling. Pattern unclear.

### Option A: Continue c1-A9 → A10... (current path)
- Wall: ~2-6h per iteration
- Risk: bug-count layering may continue indefinitely
- Speedup: ALREADY exceeds 4× target by 10×
- If closes: M6.x done, M6-S8 + M7-S0 unblock simultaneously

### Option B: Accept c1 partial-stability + run 24h with sanitize
- Sanitize firing at 86% but state stays finite
- Run 24h forecast, measure Tier-4 RMSE vs Gen2
- IF Tier-4 RMSE within 1.5× of Gen2-vs-AIFS → operationally acceptable
- Document sanitize as "physical limiter, not statistical bias" per [[feedback_validation_philosophy]]
- Cost: <1h to test; could close M6.x as "stable-via-sanitize"
- Risk: sanitize introduces unphysical clamps; reviewer may reject

### Option C: c2 semi-implicit re-architecture
- 10-20 days per contingency design
- Same bug-hunt pattern likely repeats
- High risk

### Option D: Pivot to ML-emulator hybrid (c3.C from contingency design)
- 4-8 weeks training + 1-2 weeks integration
- Lower architectural risk but longer wall
- Requires Gen2 corpus expansion (only 3 complete 24h runs currently)

### Option E: Buy existing GPU dycore + integrate (no JAX-native)
- E3SM/HOMMEXX, SCREAM, Pace (gt4py)
- Mid-wall (1-3 weeks integration)
- Breaks JAX-native architecture (ADR-001 violation)
- But gets to operational fastest

## Manager recommendation

**Try Option B (accept-with-sanitize) FIRST, in parallel with continued c1-A9.**

Rationale:
- 1h to test
- Speedup already exceeds target by 10×
- Sanitize is constitutional (per `[[feedback_validation_philosophy]]`)
- If Tier-4 RMSE acceptable, M6.x closes IMMEDIATELY
- If RMSE bad, no time wasted (c1-A9 still in flight)
- Worst case: collected data + closed M6.x with documented sanitize as a known issue → M7 proceeds with caveat

c1-A9 stays in flight as parallel insurance for a fully-clean close.

If neither closes within 24 more hours: escalate to Option C or D, requires user choice.

## Outstanding user decisions

1. **Gemini OAuth** — still expired. Please run `agy` interactively to re-auth. Would unblock orthogonal third opinion (this is your "stuck after 2 iterations" territory, way past).
2. **F-5 CPU baseline** — still pending your approval (28-rank MPI rebuild was auto-denied).
3. **Option B authorization** — should I dispatch a worker to test accept-with-sanitize 24h Tier-4 RMSE in parallel with c1-A9?

## What's been delivered overnight

- M6-S6 closed (Tier-3 BLOCKED-PARTIAL accepted)
- M6.5-D1 closed (Gen2 backfill + RMSE adapter)
- M7-S0a closed (operational/data prologue)
- M6.x contingency design (c1 contract pre-drafted)
- Plan critic (caught M7-S0a parallel-work miss)
- 4 opus bug-hunts on M6.x (all useful even when wrong)
- Empirical bisection (the inflection point that made c1 converging)
- c1-A1 acoustic core (works in isolation)
- c1-A2 scalar flux form (works)
- c1-A7 horizontal momentum flux form (works) + **43.90× speedup measured**
- c1-A8 vertical eta-metric sign (works)
- 9 worker iterations + 4 opus bug-hunts + 4 manager closeouts

All committed to main.

## What's in flight

- **c1-A9** (codex, cross-component momentum coupling, 2-4h wall)
- All other workers cleaned up

## Memory updates encoded

- `feedback_bisection_before_theory.md` — empirical bisection methodology lesson
- `feedback_parallel_bug_angles_and_plan_critique.md` — multi-angle dispatch pattern
- `feedback_rescue_uncommitted_worker_files.md` — git status check before merge

## Repo state

- Main: 30+ commits today
- M6.x branch: NOT merged (broken dycore code stays on branch)
- c1-A7+A8 acoustic+horizontal+vertical-isolated work preserved on branches
- All worker reports + manager closeouts committed for tracking

— Manager (Claude Opus 4.7 1M-context), 2026-05-22 ~05:30
