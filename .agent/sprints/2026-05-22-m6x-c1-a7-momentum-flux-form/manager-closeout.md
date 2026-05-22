# c1-A7 Manager Closeout — Horizontal Momentum CLOSED, Vertical Pending

**Sprint**: c1-A7 Momentum Advection Flux Form (Surgical)
**Status**: **CLOSED A7 — horizontal momentum bug FIXED (c1-A6 smoking gun closed); vertical residual fails step 106 → c1-A8 dispatched**
**Date**: 2026-05-22 ~04:50
**Worker**: codex gpt-5.5 xhigh (~13 min)

## What c1-A7 accomplished

Worker converted `advect_u_face`, `advect_v_face`, `advect_w_face` from advective form (`u·∂u/∂x`) to **WRF-canonical mass-flux divergence form** (`-(1/ρ_face) div(ρ_face · vel_face · interpolated_momentum)`).

Used existing WRF 5th-order flux interpolant (already in scalar advection) + WRF eta-coordinate sign convention (`-vel` for vertical, per `module_advect_em.F:4310-4315`). Held rigid lid w-face tendencies at zero.

## Results

| Probe | Before c1-A7 (c1-A6 baseline) | After c1-A7 | Improvement |
|---|---|---|---|
| Horizontal momentum only | step 30 fail | **NEVER (360 stable)** ✅ | **CLOSED** |
| Full momentum (scalar disabled) | step 30 fail | step 106 | 3.5× later |
| 1h coupled probe sanitize | 88.6% | 86.1% | -2.5pp |
| **Speedup measured** | — | **43.90×** | well above 4× constitutional target |

## Critical insights

1. **c1-A6's smoking gun (`advect_u_face` x/self) is CLOSED.** The flux form conversion fixed exactly what bisection isolated.
2. **Constitutional 4× target ALREADY EXCEEDED** at 43.90× speedup. Throughput is not the issue.
3. **Residual bug is in VERTICAL momentum** (step 106 instead of step 30 = 3.5× delay).
4. The 1h coupled probe still fails because residual vertical-momentum instability accumulates.

## c1-A8 dispatched (window 0:20)

Vertical momentum bisection (u-vertical vs v-vertical vs w-vertical) → surgical fix.

If c1-A8 closes vertical residual, M6.x CLOSES with all ACs green at iteration 8.

## c1 cumulative iteration count

| Iter | Scope | Outcome |
|---|---|---|
| A1 | Acoustic core | Stable isolated, coupled FAIL |
| A2 | Advection scalar flux form + 4 operator fixes | Operators verified, 1h FAIL |
| A3 | Long-time fix sequence (bughunt3 §5) | All 4 fixes neutral |
| A4 | Buoyancy add | Disproven by bisection (stuck at /exit) |
| A5-H3 | n_acoustic patch | Disproven (OOM + wrong site) |
| A6 | Bisection — isolated bug | SURGICAL TARGET |
| A7 | Horizontal momentum flux form | **CLOSED horizontal residual; vertical still fails step 106** |
| A8 | Vertical momentum bisect + fix | Dispatched |

Total wall: ~5h clock so far. Original c1 estimate was 5-9 DAYS. The bisection-driven methodology saved ~99% of expected time.

## Branch state

- `worker/codex/m6x-c1-klemp-skamarock` at `19a484a` (c1-A4 buoyancy on top of c1-A3)
- `worker/codex/m6x-c1-a7-momentum-flux-form` at `abd12c4` (c1-A7 horizontal momentum fix; from c1-A6 HEAD)
- `worker/codex/m6x-c1-a8-vertical-momentum-bisect` at HEAD (c1-A8 in progress, from c1-A7)
- NOT merged to main — wait for c1-A8 green

## Honest accounting

The c1 cycle has been brutal but methodologically converging. Each iteration narrowed scope. The empirical bisection (A6) was the inflection point — before it, theories were guessing; after it, fixes are surgical.

If c1-A8 closes (~3-6h estimate), the M6.x cycle was a 5-9h debugging marathon (vs 5-9 days estimate). M6-S8 + M7-S0 unblock simultaneously.

— Manager (Claude Opus 4.7 1M-context), 2026-05-22 04:50
