# c1-A8 Manager Closeout — Vertical Eta Metric CLOSED, Cross-Coupling Pending

**Sprint**: c1-A8 Vertical Momentum Bisect + Surgical Fix
**Status**: **CLOSED A8 — vertical w-momentum eta-metric sign fixed; cross-component coupling residual → c1-A9**
**Date**: 2026-05-22 ~05:25
**Worker**: codex gpt-5.5 xhigh

## What c1-A8 fixed

Vertical momentum flux divergence metric SIGN was wrong for this codebase's positive geometric dz.
- WRF uses `znw[k+1]-znw[k]` (znw decreases upward → dnw negative); divergence is `-rdzw·(vflux[k+1]-vflux[k])`
- c1 uses positive geometric dz; needed OPPOSITE sign

Cited WRF: `module_advect_em.F` vertical momentum branches + `module_initialize_real.F:3733-3734` (dnw, rdnw definitions).

## Results

| Probe | Before c1-A8 | After c1-A8 |
|---|---|---|
| Vertical w-momentum only | step 158 ✗ | **NEVER (360 stable)** ✅ |
| Vertical u-momentum only | stable | stable ✅ |
| Vertical v-momentum only | stable | stable ✅ |
| Vertical momentum only | step 158 | **NEVER (stable)** ✅ |
| u+v no w | stable | stable |
| u+w | (pair not run before) | step ~193 ✗ |
| v+w | (pair not run before) | step ~197 ✗ |
| Full momentum (scalar disabled) | step 106 | step 188 |
| 1h coupled sanitize | 86.1% | 86.94% (essentially unchanged) |
| **Speedup measured** | 43.90× | 44.33× |

## Why coupled still fails

Cross-component momentum coupling. `u alone` stable, `w alone` stable, but `u+w` together fails. Classic reflexivity bug: what u-face sees from w should match what w-face sees from u.

## c1-A9 dispatched (window 0:21)

Cross-component bisection (u←w vs w←u) + reflexivity audit + surgical fix.

## Branch state

- `worker/codex/m6x-c1-a8-vertical-momentum-bisect` at `17ac067` (vertical eta sign fix)
- c1-A9 worktree branched from this

## Strategic position

9 c1 iterations. ~7h clock so far. Bisection methodology continues to deliver surgical evidence. **Speedup is already proven (>40× vs 4× target).** Stability incrementally improving but not closed.

c1-A9 may close it. May not. Decision package written at `MORNING-REPORT-2026-05-22.md` for user wake-up.

— Manager (Claude Opus 4.7 1M-context), 2026-05-22 ~05:25
