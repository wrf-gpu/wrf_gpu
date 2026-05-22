# M6.x Empirical Bisection — Manager Closeout

**Sprint**: Empirical bisection (systematic component disable)
**Status**: **CLOSED — DEFINITIVE finding: bug is in advection, c1-A6 internals-bisect dispatched**
**Date**: 2026-05-22 ~04:05
**Worker**: codex gpt-5.5 xhigh (~13 min)

## DEFINITIVE FINDING

| Configuration | first_nonfinite_step (sanitize-disabled, 1h probe) |
|---|---|
| Full baseline | step 25 |
| No boundary | step 25 (boundary NOT cause) |
| No physics | step 26 (physics NOT cause) |
| Dycore only | step 25 |
| **Acoustic only** | **NEVER (360 steps stable)** ✓ |
| **Advection only** | **step 30** ✗ |
| mu-continuity only | NEVER ✓ |

**THE BUG IS IN ADVECTION.** Not acoustic, not boundary, not physics, not mu-continuity.

## What this ELIMINATES

- bug-hunt #1 (PH g, asymmetric damping): WRONG — acoustic alone is stable
- bug-hunt #2 (buoyancy, diagnostic-p, ρ factor): WRONG — acoustic alone is stable
- bug-hunt #3 (periodic wrap in stencils, smdiv, ph advection): WRONG — c1-A3 changed acoustic/advection but acoustic-only still stable, meaning acoustic changes weren't load-bearing AND advection changes (c1-A2) introduced the bug
- bug-hunt #4 H1 (split-physics): NOT IT — no physics still fails at step 25
- bug-hunt #4 H2 (boundary subset): NOT IT — no boundary still fails at step 25
- bug-hunt #4 H3 (n_acoustic=86): NOT IT — acoustic-only is stable (n_acoustic value doesn't matter)
- c1-A4 (buoyancy add) in flight: WILL FAIL — acoustic alone is stable so buoyancy isn't the bug
- c1-A5-H3 (n_acoustic patch) in flight: WILL FAIL — see above

## What c1-A2 changed in advection that may be the bug

c1-A2 rewrote `advect_mass_scalar` to mass-conservative flux form. Unit tests pass at 1e-10. But the coupled probe shows advection-only goes nonfinite at step 30.

Likely candidates:
- `advect_mass_scalar` + `derivative5_upwind` (scalar advection)
- `advect_u_face`, `advect_v_face`, `advect_w_face` (face-velocity advection)
- `_periodic_flux5_faces` (or its c1-A3 replacement)

## c1-A6 dispatched (window 0:18)

**Per bughunt #4 anti-pattern warning: NO bundling. Isolate-only, no fix.**

c1-A6 bisects ADVECTION internals: scalar vs momentum, horizontal vs vertical, per-field. Goal: find the single operator. Then c1-A7 surgically fixes only that operator.

## c1-A4 and c1-A5-H3 status

Both running. Both pursuing disproven hypotheses (acoustic was proven stable). Letting them finish naturally to avoid interrupt cost — they'll report failure and self-rectify.

## Strategic position

Iteration #7 of M6.x debugging (A1 + A2 + c1-A1 + c1-A2 + c1-A3 + c1-A4 + c1-A5-H3 + bisection). The empirical bisection is the highest-information-per-dollar sprint of the entire M6.x cycle. It eliminated 7 wrong hypotheses with 7 cheap probes.

c1-A6 will isolate the advection operator (~2-4h). Then c1-A7 (~30min to 2h) surgically fixes it.

## Decision logic

After c1-A6:
- If specific operator isolated → c1-A7 surgical fix → M6.x closes
- If c1-A6 also inconclusive → escalate to user (suggests advection.py is structurally wrong, may need to revert c1-A2's rewrite)

## Honest accounting

The bisection methodology PAID OFF. After 3 wrong bug-hunts and 5 failed worker iterations, ONE bisection sprint gave a definitive answer.

**Lesson for future M6 debugging**: empirical bisection BEFORE theory. Cheap probes that disable components are more informative than hypothesis-driven code audits.

— Manager (Claude Opus 4.7 1M-context), 2026-05-22 04:05
