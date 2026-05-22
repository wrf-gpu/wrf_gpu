# Bug-Hunt #4 Meta-Analysis — Manager Closeout

**Sprint**: Bug-Hunt #4 Meta-Analysis (what 3 previous bug-hunts missed)
**Status**: **CLOSED — Opus delivered 3 composition-level hypotheses; H3 discriminator dispatched**
**Date**: 2026-05-22 ~04:00
**Reviewer**: Claude Opus 4.7 xhigh (1M)
**Wall**: ~50 min

## Meta-finding

**All 3 previous bug-hunts framed the system as "the dycore is wrong" — they all looked INSIDE the operators.** Bug-hunt #4 audited what was OUTSIDE: composition, setup, contracts. Found 3 NEW hypotheses no previous bug-hunt opened.

## 3 NEW meta-level hypotheses

### H1: Split-physics architecture (HIGH)
- `Tendencies.zeros(grid)` once at init, NEVER updated (`coupling/driver.py:124`)
- Physics tendencies NEVER enter RK3 RHS (`dynamics/rk3.py:40-41`)
- Physics adapters DIRECTLY replace state (`physics_couplers.py:182-365`)
- None touch p, pb, ph, mu → column leaves hydrostatic balance EVERY step
- Next acoustic substep sees inconsistent `p - pb` → runaway

### H2: Boundary replay forces only 6/42 leaves (MEDIUM-HIGH)
- `apply_lateral_boundaries` replays u, v, theta, qv, ph, mu only
- p, pb, w left drifting at boundary
- Boundary p inconsistent with replayed theta/ph/mu → spurious gradient pushes inward

### H3: `n_acoustic=86` via `_flat_dz` fallback misuse (HIGHEST, lowest-cost discriminator)
- `required_n_acoustic` (`dynamics/acoustic.py:242-249`) uses `_flat_dz`
- `_flat_dz` (lines 86-90) is documented as **TEST-ONLY** fallback for analytic states
- Production Gen2 runs hit it → wrong CFL → n_acoustic auto-promoted from 2 to 86
- 92,880 substeps/hour × per-substep noise = 9.3% drift/hour
- **Explains "step-45 nonfinite is the same regardless of operator fix"** signature exactly

## CRITICAL anti-pattern warning

Bug-hunt #4 §4: **"Do NOT dispatch c1-A5 to apply all three fixes at once — that's the same anti-pattern as bug-hunts #2 and #3."** Discriminator value requires ONE-at-a-time changes. Manager adopts this.

## Manager dispatch

Per bug-hunt #4 §4 recommendation:
- **c1-A5-H3 discriminator** dispatched (window 0:17): 1-line patch to `rk3.py:49` to disable auto-promotion, 1h probe, observe
- Already in flight: c1-A4 (buoyancy add — bughunt2 Hyp A unresolved) + empirical bisection (covers H1 and H2 discriminators implicitly via component disable)

## Decision logic

After c1-A5-H3 reports:
- If sanitize <50% → H3 is dominant; c1-A6 properly fixes (use `required_n_acoustic_for_state` or similar)
- If sanitize >70% → H3 not dominant; bisection results will tell H1 vs H2 vs something else
- If sanitize ~0% → H3 fully explains; M6.x closes immediately

After bisection reports:
- Phase 1 reveals if dycore alone vs dycore+physics vs dycore+boundary is the failure mode
- Phase 2-4 narrow further

After c1-A4 reports (buoyancy add):
- Independent test of whether buoyancy was truly missing

## Why bug-hunt #4 found what others missed

1. Different framing: "system composition" vs "operator stencil"
2. Read `coupling/driver.py` + `physics_couplers.py` (NONE of 3 previous bug-hunts opened these deeply)
3. Recognized "step-45 every time regardless of fix" as evidence of per-substep accumulator (not per-step operator error)

## Critical-path state

After c1-A5-H3 (~30-60 min) + bisection (~2-4h) + c1-A4 (~2-4h) all report, we'll have:
- H3 isolated yes/no
- H1 isolated yes/no (via bisection Phase 1)
- H2 isolated yes/no (via bisection Phase 1)
- Buoyancy isolated yes/no (via c1-A4)

That's 4 independent data points. At LEAST one should be definitive. If all 4 inconclusive → user escalation territory.

## Strategic position

This is iteration #6 of M6.x debugging. The pattern is converging:
- Operator-level theories: 3 wrong → composition-level theory under test
- 2/3 codex + 1/3 opus capacity used; 1 codex + 2 opus spare

If bug-hunt #4 is right that the bug is in setup/composition, M6.x closes very soon. If wrong, user-level architectural decision required.

— Manager (Claude Opus 4.7 1M-context), 2026-05-22 04:00
