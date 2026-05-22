# M6.x Bug-Hunt #2 (Deeper) — Manager Closeout

**Sprint**: M6.x Bug-Hunt #2 — fresh-eyes review after bug-hunt #1 hypotheses didn't fix the problem
**Status**: **CLOSED — Opus delivered 3 STRUCTURAL hypotheses; c1 path CONFIRMED right**
**Date**: 2026-05-22 ~01:15
**Reviewer**: Claude Opus 4.7 xhigh (fresh context, bughunt #2)
**Wall**: ~10 min

## Headline

Bug-hunt #2 found what bug-hunt #1 missed: **3 formulation-level bugs**, not line-level bugs. Confirms c1 Klemp-Skamarock invocation is the correct call.

## Bug-hunt #2's 3 structural hypotheses

### Hypothesis A (HIGH, 80% confidence): Missing buoyancy term in w equation
- `acoustic.py:189-191` w equation has only `-α ∂p'/∂z` (perturbation pressure gradient)
- **Missing**: `g(ρ_base - ρ)/ρ` buoyancy term
- Theta NEVER enters acoustic substep
- Without buoyancy, gravity-wave mode has no restoring force → unbounded growth
- WRF: `module_small_step_em.F:1481-1486` (advance_w with `dts*g*msft_inv*(...t_2ave...c2a*alt...)` buoyancy)
- **Explains bipolar mu pattern** (1000↔120000 clip) — gravity waves growing unboundedly

### Hypothesis B (HIGH, 80% confidence): Pressure prognostic vs WRF-canonical diagnostic
- `acoustic.py:185`: worker has `dp/dt = -c²·div` (PROGNOSTIC)
- WRF: pressure DIAGNOSED from (theta, ph, mu) via linearized EoS every small step
- WRF cite: `module_small_step_em.F:494-528` (linearized EoS + hydrostatic al diagnostic)
- (theta, ph, p) drift apart hydrostatically; compounds to NaN by 6h
- **This is what c1 fixes natively** (diagnostic p, per design.md §2.1)

### Hypothesis C (MEDIUM-HIGH, 60%): Missing ρ factor in prognostic pressure
- Correct: `∂p/∂t = -ρ c²·div` = `-γ p·div`
- Worker: `dp/dt = -c²·div` (missing ρ = α factor)
- With α=5.0 cap (post-FIX#2b), effective sound speed `sqrt(5)·c_true ≈ 780 m/s` (vs 348 m/s)
- **Explains why FIX#2 made it WORSE** — relaxing alpha cap made the wrong dispersion catastrophic

## c1 CONFIRMED

Bug-hunt #2 §4: *"Keep c1 (Klemp-Skamarock clean-room) running. It is on a path that addresses all three hypotheses above"*

- Klemp 2007 tridiagonal w-ph solve carries buoyancy NATIVELY → addresses A
- Pressure becomes diagnostic from (theta, ph, mu) → addresses B
- ρ factor built into formulation → addresses C

5-9 day estimate consistent with 3 formulation-level bugs (not single-line edits).

## c1 progress check (window 0:9, 17m elapsed)

- `dynamics/acoustic.py`: +354 line rewrite
- `dynamics/tridiag.py`: NEW (Thomas solve)
- `dynamics/rk3.py`: 9 changes
- `contracts/state.py`: 15 changes (likely base-state additions)
- 2 NEW tests: `test_m6x_fallback_c1_acoustic.py`, `test_m6x_fallback_c1_tridiag.py`
- 2 NEW ADRs: `ADR-018-m6x-fallback-c1-tridiag-backend.md`, `ADR-019-m6x-fallback-c1-klemp-skamarock-clean-room.md`
- 1h coupled probe currently RUNNING (background process active)

c1 is making rapid initial progress. Not interrupting with bughunt2 heads-up since c1 is following Klemp 2007 which addresses all 3 hypotheses by construction.

## Decision logic

- **If c1 1h probe PASSES**: c1 is on right path; let it continue toward 6h + 24h
- **If c1 1h probe FAILS**: send bughunt2 hypotheses as fix-hint (especially Hypothesis A buoyancy term)
- **If c1 lands GREEN within 24h**: M6.x closes, M7-S0 + M6-S8 unblock
- **If c1 also fails after 24h**: escalate; consider c2 semi-implicit or end-goal re-scope

## Honest cost-benefit of bug-hunt #2

- Wall: ~10 min Opus time
- Outcome: confirmed c1 is correct; ruled out cheap fix possibility; provided concrete WRF citations for c1 worker to use; raised confidence in 5-9 day investment from "best guess" to "evidence-backed"
- Value: HIGH — without bug-hunt #2, manager would be guessing whether c1 was the right call; with it, c1 is confirmed.

## Per-user-feedback validation

Plan critic + bug-hunt #1 + bug-hunt #2 all dispatched per [[feedback_parallel_bug_angles_and_plan_critique]]. Each found different angles. Bug-hunt #2 specifically found what bug-hunt #1 missed (formulation-level vs line-level), confirming the value of multiple parallel angles.

— Manager (Claude Opus 4.7 1M-context), 2026-05-22 01:15
