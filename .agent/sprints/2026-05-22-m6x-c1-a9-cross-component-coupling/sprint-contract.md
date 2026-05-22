# Sprint Contract — c1-A9 Cross-Component Momentum Coupling Bisect

**Sprint ID**: `2026-05-22-m6x-c1-a9-cross-component-coupling`
**Created**: 2026-05-22 ~05:25
**Status**: ACTIVE
**Trigger**: c1-A8 vertical eta-metric sign fix CLOSED isolated vertical-w bug (stable 360). But cross-component pair probes (u+w, v+w) still fail at step 193-197. Cross-component momentum coupling has a residual bug.
**Branch**: from `worker/codex/m6x-c1-a8-vertical-momentum-bisect` HEAD `17ac067`
**Worktree**: `/tmp/wrf_gpu2_c1_a9`

## Objective

Bisect within cross-component momentum coupling. Isolate specific (u+w) interaction term, apply surgical fix per WRF.

## Bisection plan

Pair probes already done (per c1-A8 worker report):
- u alone, v alone, w alone: each stable
- u+v: stable
- u+w, v+w: fail step 193-197

So the bug is in u-w or v-w cross-coupling. Specifically: w's effect on u-face, or u's effect on w-face.

### Phase 1: per-direction cross-coupling probes
1. u-face advection vertical metric only (w advecting u vertically): how it behaves alone
2. w-face advection horizontal metric only (u advecting w horizontally): how it behaves alone
3. Each one disabled in turn within u+w configuration

### Phase 2: surgical fix per WRF
Likely fixes:
- Wrong staggering: w at u-face vs u at w-face — interpolation may be using wrong convention
- Missing 0.5 factor at staggered face
- Sign issue at cross-stagger

WRF reference: `dyn_em/module_advect_em.F` — the advect_u routine's vertical-w term should match advect_w's horizontal-u term reflexively.

## Acceptance

- AC1: cross-component pair probes (u+w, v+w) stable through 360 steps
- AC2: 1h coupled probe sanitize <5%
- AC3: 24h coupled probe finite, theta in [200,350], mu in [5000,110000]
- AC4: Speedup ≥4× (already 43.90× measured at c1-A7)
- AC5: ADR-007 → PASS-with-evidence
- AC6: ADR-019 amended

## File ownership

- MODIFY: `src/gpuwrf/dynamics/advection.py` (cross-coupling terms in advect_u/v/w_face)
- MODIFY: tests
- DON'T TOUCH: acoustic, tridiag, rk3, state, physics, boundary_apply
- c1-A7 horizontal momentum + c1-A8 vertical eta sign FROZEN

## Hard rules

- Bisect FIRST, fix SECOND
- ONE fix per probe
- Cite WRF source
- `git add . && git commit && git push` BEFORE `/exit`
- `/exit` slash-command

## Dispatch

- Worker: codex gpt-5.5 xhigh
- Wall: 2-4h

## End-goal

If c1-A9 closes, M6.x lands GREEN. Total c1 cycle: ~9-10h (still vastly under 5-9 day estimate).
If c1-A9 also fails: USER ESCALATION required (Gemini opinion may have landed by then).
