# M6.x Bug-Hunt #3 (Long-Time Integration) — Manager Closeout

**Sprint**: Bug-Hunt #3 — Long-time integration accumulator hunt
**Status**: **CLOSED — Opus delivered 3 long-time hypotheses with 5-step fix sequence; c1-A3 dispatched**
**Date**: 2026-05-22 ~03:10
**Reviewer**: Claude Opus 4.7 xhigh
**Wall**: ~25 min

## Headline

Bug-hunt #3 found 3 long-time accumulators that explain why 18-step probe is clean but 1h probe is 88.6% sanitize:

### Hypothesis A (HIGH, 75%): Periodic-wrap stencils on non-periodic d02 grid
- **CRITICAL FINDING**: `contracts/halo.py:28-32` `apply_halo` is a NO-OP
- 7 dycore stencils still use `jnp.roll` PERIODIC on limited-area d02 grid:
  - `acoustic.py:139-147` `_grad_x_to_u`, `_grad_y_to_v`
  - `acoustic.py:162-170` `_mass_to_u_face_2d`, `_mass_to_v_face_2d`
  - `advection.py:218-229` `_periodic_flux5_faces`
  - `advection.py:303-312` `_mass_to_u_face`, `_mass_to_v_face`
- **c1-A2 fixed the SAME bug in `physics_couplers.py` but missed all 7 dycore call sites**
- 92,880 substeps/hr × periodic wrap → coherent standing wave growth (subthreshold at 18 steps)

### Hypothesis B (HIGH): Inconsistent dz floor across modules
- `acoustic.py:_layer_thickness_m` NO floor for positive dz
- `advection.py:_dz_from_state` floors at 1.0 m
- When ph drifts and dz collapses to <1m, the tridiag coefficients (scaling as 1/dz²) explode → NaN ignition spark

### Hypothesis C (MEDIUM-HIGH): Missing Klemp §3d + missing ph horizontal advection
- C1: no `smdiv·(p - p_prev)` divergence damper → forward-backward acoustic is neutrally stable, drifts to growing in 92k substeps
- C2: `ph` evolved only by `+g·dt·w` inside acoustic; never horizontally advected → interior ph drifts; boundary pinned by Gen2 → boundary-interior mismatch becomes a step-discontinuity at every boundary apply

## Manager dispatched c1-A3 (window 0:13)

**5-step fix sequence per bughunt3 §5** (cheapest first, biggest leverage):
1. Pre-flight: sanitize-disable 1h diagnostic (sanitize-driver vs sanitize-catcher discriminator)
2. **FIX #A**: replace `jnp.roll` in 7 dycore stencils with edge-mirror (the c1-A2 pattern) — expected ≥10× drop in nonfinite, ≥5× drop in firing rate
3. **FIX #B**: 1-line dz floor in `acoustic.py:_layer_thickness_m`
4. **FIX #C1**: add Klemp §3d smdiv divergence damper (`smdiv=0.1` per `module_small_step_em.F:562`)
5. **FIX #C2**: add ph to `compute_advection_tendencies` per `module_em.F:1292 advect_ph_implicit`

Wall: 4-8h.

## Bug-hunt #3 also flags (uncertainty)

- **70% confident buoyancy still missing** in `_vertical_implicit_w` (bug-hunt #2 Hyp A may not be fully closed)
- Constant reference `α` and `ρc²` instead of state-dependent → up to 5× wrong at column extremes
- μ-coupling outside acoustic substep (bug-hunt #1 Hyp #3 structural decoupling persists)

These are tagged for c1-A4 or c2 escalation if c1-A3 fails to close.

## Strategic position

c1 iteration sequence:
- A1 (acoustic core): 48 min
- A2 (advection/coupling operators): 40 min
- A3 (long-time integration): in flight, 4-8h estimated

Total c1 wall: <12h so far (vs 5-9 day original estimate). Bug-hunt #1 + #2 + #3 each found different bugs at different abstraction levels (line-level → operator-level → long-time-integration-level). Parallel-angles methodology PAID OFF.

## Decision logic

- If c1-A3 1h probe PASSES → 6h + 24h proof → M6.x closes GREEN
- If c1-A3 FIX #A alone doesn't drop nonfinite ≥10× → suspect bughunt3's hypothesis priority wrong → escalate
- If c1-A3 all 4 fixes still fails → user-level decision (re-open buoyancy, escalate to c2, end-goal pivot)

## Gemini still blocked

OAuth expired; bughunt3's analysis didn't get Gemini's orthogonal angle. User `agy` re-login needed for next iteration if c1-A3 fails.

— Manager (Claude Opus 4.7 1M-context), 2026-05-22 03:10
