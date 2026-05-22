# c1-A2 Manager Closeout — All 4 Fixes Landed; Long-Time Integration Still FAIL

**Sprint**: c1-A2 Advection + Coupling Fix-Hint Application
**Status**: **CLOSED A2 — All 4 fixes verified individually; 1h coupled probe still FAIL → bug-hunt #3 dispatched**
**Date**: 2026-05-22 ~02:35
**Worker**: codex gpt-5.5 xhigh (~40 min)

## Headline

All 4 named fix targets LANDED + verified individually:

| FIX | Test/Probe | Before | After |
|---|---|---|---|
| #3 mass-conservation | `test_mass_scalar_advection_is_conservative_for_constant_velocity` | 8.43e-05 (6 OOM violation) | **PASS at 1e-10** |
| #1 per-layer dz | 18-step coupled probe sanitize | 1/18 | 0/18 |
| #2 non-periodic interpolation | 18-step coupled probe sanitize | 0/18 | 0/18 |
| #4 perturbation-pressure advection | `test_pressure_advection_transports_perturbation_not_static_base_state` | n/a | PASS |

18 tests pass; all c1 frozen files verified UNTOUCHED.

## BUT: 1h coupled probe FAIL

```
fired_steps=319, step_firing_rate=88.6%, nonfinite_count=793,817,606
clip_count=274,320,190, final theta=[150,550] K, u/v=150 m/s, w=50 m/s, p=[1000,120000] Pa
```

Operators are correct (proven by 18-step + unit tests). Bug is in **long-time integration accumulation** — emerges between minute-2 and minute-60 of simulated time.

## Manager dispatched bug-hunt #3 (window 0:12)

Focus areas:
- **Sanitize as driver**: does even 1 clip event per step inject a shock that compounds?
- **Physics couplers timing**: Thompson/MYNN/RRTMG outputs may be inconsistent with c1's diagnostic-p
- **Boundary replay timing**: Gen2 forcing has CPU-WRF (p, ph, mu); interior has c1-GPU formulation; interface mismatch may drive long-time gravity-wave generation
- **Halo width vs 5th-order advection stencil**: needs 3-cell halo; may be 2
- **Conservation drift**: integral of mu, total water, total energy over 360 steps
- **Time-step order**: dycore → physics → boundary; are any inputs stale?

## Strategic position

This is **iteration 5 on M6.x dycore**:
- A1: option a-narrowed (FAIL)
- A2 (bughunt #1): FAIL — bug-hunt's hypothesis wrong direction
- c1-A1: acoustic core works in isolation
- c1-A2 (bughunt #2 + worker self-diagnosis): operators fixed but 1h fails
- Bug-hunt #3: now searching long-time integration

Per user feedback `[[feedback_parallel_bug_angles_and_plan_critique]]`: "ask gemini if things get stuck and neither gpt nor opus can fix a core bug after 2 iterations" → **Gemini OAuth still expired**; cannot dispatch.

## Decision logic

- If bug-hunt #3 finds a specific bug → c1-A3 codex with hint
- If bug-hunt #3 inconclusive → user-level escalation needed (c2 semi-implicit, sanitize-disable diagnostic, or end-goal architectural pivot)
- Gemini orthogonal opinion blocked on OAuth (need user `agy` re-login)

## Branch state

- `worker/codex/m6x-c1-klemp-skamarock` at `7e3301f` (c1-A2 fixes committed by worker)
- NOT merged to main — broken state
- worker-report + sprint-contract copied to main for tracking

## What worked (for the record)

The plan critic → bug-hunt #1 → bug-hunt #2 → c1 dispatch → c1-A1 (48min vs 5-9d) → c1-A2 (40min vs 4-8h) sequence is faster than expected. Bug-hunt #3 should land within the hour. Even with a 5-iteration M6.x sequence, total wall is ~5h of clock time — fast cycle.

The remaining unknown: is there a fixable bug in long-time integration, or does the c1 architecture have a fundamental limitation that needs c2?

— Manager (Claude Opus 4.7 1M-context), 2026-05-22 02:35
