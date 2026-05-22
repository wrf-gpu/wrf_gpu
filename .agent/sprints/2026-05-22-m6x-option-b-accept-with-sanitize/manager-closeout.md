# Option B Manager Closeout — Accept-with-Sanitize FAILS Tier-4 RMSE Gate

**Sprint**: M6.x Option B — Accept-with-Sanitize 24h Tier-4 RMSE Test
**Status**: **CLOSED — FAIL by 14× (ratio 21.26 vs ≤1.5 PASS threshold)**
**Date**: 2026-05-22 ~06:45
**Worker**: codex gpt-5.5 xhigh (~52 min)

## Headline

**Option B verdict: FAIL.** max_ratio_gpu_gen2 / gen2_aifs RMSE = **21.26×** (need ≤ 1.5×, partial ≤ 2.0).

The "accept-with-sanitize" path does NOT produce operationally useful forecasts. State saturates at sanitize bounds (theta=[150,550], mu=[1000,120000], u/v=150 m/s, w=50 m/s) + final state contains nonfinite leaves.

## Hard data

| Metric | Value |
|---|---|
| max_ratio_gpu_gen2 / gen2_aifs | **21.26** |
| pass_threshold | 1.5 |
| partial_threshold | 2.0 |
| final all_state_leaves_finite | **false** |
| final theta range | [150, 550] K (clipped both bounds) |
| final mu range | [1000, 120000] Pa (clipped both bounds) |
| final u/v abs max | 150 m/s (clipped) |
| final w abs max | 50 m/s (clipped) |

The forecast is fundamentally broken; sanitize is just preventing NaN propagation, not producing physically valid output.

## Eliminated paths

Per `[[feedback_validation_philosophy]]` ("operational RMSE > bitwise parity"), Option B was the cheap shortcut to close M6.x. It FAILED definitively. So:
- M6.x cannot close on c1-A8 HEAD with sanitize as the safety net
- The cross-component cumulative drift overwhelms the sanitize clamp by step ~200 simulated time
- 24h forecast has 6+ hours of fully saturated state

## What remains (all need user authorization)

### Option A: Continue c1-A10 with map-factor extension
- Extends State pytree (msfvy/msfvx fields) + GridSpec
- **BREAKS ADR-002 baseline** — requires user authorization
- Wall: 4-8h surgical work + verification probes
- Risk: even with map-factors, may still have residual layers

### Option C: c2 semi-implicit re-architecture
- 10-20 days per contingency design
- Same bug-hunt pattern may repeat
- High architectural risk

### Option D: ML-emulator hybrid (c3.C)
- 4-8 weeks training + 1-2 weeks integration
- Requires Gen2 corpus expansion (only 3 complete 24h runs)
- Lower architectural risk; longer wall

### Option E: Buy existing GPU dycore (E3SM/HOMMEXX, SCREAM, Pace)
- Breaks JAX-native architecture (ADR-001)
- 1-3 weeks integration
- Gets to operational fastest

### Option F: Accept that 44.33× speedup is the deliverable; pivot M6 to throughput-only closeout
- M6 milestone closes as "constitutional 4× target achieved at 44× measured on M4 reduced dycore; full WRF-canonical dycore deferred to M7+"
- Forecast operational capability moved to future milestone
- Lowest-cost close but doesn't unblock M7 operational target

## Manager non-recommendation

I have exhausted all autonomous options. **User decision required.**

The c1 dycore methodology has converged on map-factor support as the residual blocker. Adding map-factors is a 4-8h surgical sprint but breaks the ADR-002 frozen baseline. User must authorize.

Alternatively, the constitutional target IS achieved (44.33×); the question is whether M6 closes as "throughput-only" or requires "operational stability" too.

## Sleep state

- Option B worker exit'd cleanly with commit `40216d7`
- All probe artifacts committed
- 9 c1 iterations + 4 bug-hunts + plan critic all on main
- MORNING-REPORT-2026-05-22.md decision package updated
- Workers active: NONE — all paths exhausted

## Branch state

- `worker/codex/m6x-option-b-accept-with-sanitize` at `40216d7` (RMSE artifact + script committed; NOT merged to main since FAIL)

— Manager (Claude Opus 4.7 1M-context), 2026-05-22 ~06:45
