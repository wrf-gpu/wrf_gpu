# c1-A6 Advection-Internals Bisection — Manager Closeout

**Sprint**: c1-A6 advection-internals bisect
**Status**: **CLOSED — bug isolated to `advect_u_face` x/self term; c1-A7 surgical fix dispatched**
**Date**: 2026-05-22 ~04:35
**Worker**: codex gpt-5.5 xhigh (~30 min)

## The bug

**`src/gpuwrf/dynamics/advection.py:381`** in `advect_u_face`:
```python
state.u * derivative5_upwind(state.u, state.u, _dx(grid), axis=2)
```

This is **advective form** `u·∂u/∂x` — NON-CONSERVATIVE for momentum. c1-A2 rewrote scalar advection to mass-conservative flux form but left momentum in advective form. Mismatch is the bug.

## Bisection evidence

| Probe | first_nonfinite_step |
|---|---|
| advection-only | 30 (proven by prior bisection) |
| scalar-only | 189 (much later, different bug) |
| **momentum-only** | **30** ✗ (dominant) |
| horizontal momentum-only | 30 |
| vertical momentum-only | 112 (slower) |
| **horizontal `u` only** | **30** (smallest reproducer) |
| horizontal `v` only | 41 (slower) |
| horizontal `w` only | NEVER stable |
| **`u` x/self only** | **31** (smoking gun) |
| `u` y/cross only | NEVER stable |
| `u` x+y combined | 30 |

Surgical conclusion: the bug is in `advect_u_face` x/self term. Likely also exists in `advect_v_face` and `advect_w_face` (same advective-form pattern), but `u` is the smallest reproducer.

## c1-A7 dispatched (window 0:19)

Worker: codex xhigh
Scope: convert `advect_u_face`, `advect_v_face`, `advect_w_face` from advective form to WRF-canonical flux form.
Wall: 2-4h.
Worktree: `/tmp/wrf_gpu2_c1_a7` (branched from c1-A6's bisected commit `eaeaec9`)

## Strategic position

7 c1 iterations in:
- A1: acoustic core (works in isolation)
- A2: 4 operator fixes (mass-conservation OK, but momentum advective form bug introduced/persisted)
- A3: 4 long-time fixes (all wrong direction)
- A4: buoyancy (wrong direction, but committed on c1 branch as 19a484a — bonus quality improvement)
- A5-H3: inconclusive (OOM + wrong patch site)
- A6: BISECTION ISOLATED THE BUG (this sprint)
- A7: SURGICAL FIX (dispatched)

If c1-A7 lands: M6.x closes. Total c1 wall: ~10-12h vs 5-9 day estimate.

## Lesson reinforced

**Bisection BEFORE theory.** c1-A6 cost ~30 min and gave a SURGICAL target. 3 prior opus bug-hunts (~2h total) gave 9 theories all wrong on the dominant accumulator.

The methodology lesson: when fixing 5+ wrong theories in a row, STOP THEORIZING and INSTRUMENT TO BISECT.

— Manager (Claude Opus 4.7 1M-context), 2026-05-22 04:35
