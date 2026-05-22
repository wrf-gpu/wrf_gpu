# Sprint Contract — c1-A2 Advection + Coupling Fix-Hint

**Sprint ID**: `2026-05-22-m6x-c1-a2-advection-coupling-fix`
**Created**: 2026-05-22 ~01:25
**Status**: ACTIVE
**Trigger**: c1-A1 landed Klemp-Skamarock acoustic core (60 step acoustic-only stable; 18-step coupled probe clean). 1h coupled probe FAIL — **NOT in acoustic core**. Bugs are in advection + coupling interaction with new pressure formulation. Bug-hunt #2 §2 + c1-A1 self-diagnosis converge on specific operators.

## Critical evidence

- **Acoustic-only 60-step probe: STABLE** (c1 acoustic core is correct)
- **18-step coupled probe: STABLE** (sanitize firing zero)
- **1h coupled probe: FAIL** (even with radiation disabled)
- **`test_mass_scalar_advection_is_conservative_for_constant_velocity`: FAIL** at 8.43e-05 vs 1e-10 tolerance (6 OOM violation)
- **Coupled-with-advection becomes nonfinite at step 30**, acoustic-only doesn't

## Diagnosis (from bughunt2 + c1-A1 self-report)

The bug is in advection/coupling, NOT acoustic. Specific targets:

### Target #1: `_dz_from_state` uses MEAN dz (advection.py:41-45)
- Returns mean dz across all layers as scalar passed to vertical 3rd-order upwind
- WRF eta levels: ~30 m near surface vs ~1000 m aloft
- Using mean (~300 m) → near-surface gradients 10× too weak, aloft 3× too strong
- Systematic bias in scalar advection
- **Fix**: use per-layer dz (state.ph difference / g, per-cell)

### Target #2: `_mass_to_u_face` uses `jnp.roll` PERIODIC interpolation (physics_couplers.py:98-109)
- For d02 limited-area grid, this corrupts u-face 0, u-face nx, v-face 0, v-face ny with periodic-wrap of opposite boundary values
- bug-hunt #2 noted: `apply_lateral_boundaries` overwrites these immediately after, so leak is at most one cell per step pre-overwrite
- BUT: with c1's stronger acoustic + advection more sensitive, that one-cell leak per step amplifies
- **Fix**: use non-periodic stencil (mirror or extrapolation at edges)

### Target #3: Mass-scalar advection conservation oracle fails
- `test_mass_scalar_advection_is_conservative_for_constant_velocity` at 8.43e-05 vs 1e-10
- 6 orders of magnitude violation
- **Fix**: audit `compute_advection_tendencies` or upwind operator for mass-conservation form (correct: ∂(ρq)/∂t + ∇·(ρqv) = 0, NOT ∂q/∂t + v·∇q = 0)

### Target #4: c1-A1 changed `advection.py` to advect perturbation pressure instead of total pressure
- Worker self-flagged: "reviewer should explicitly accept or reject"
- This is correct for c1's diagnostic-pressure scheme (don't transport hydrostatic base state)
- BUT: needs verification that the perturbation-pressure-advection is consistent with the rest of the c1 pressure formulation

## Acceptance

- **AC1** `_dz_from_state` uses per-layer dz; test added that exercises non-uniform dz scenario
- **AC2** `_mass_to_u_face` + `_mass_to_v_face` non-periodic; new tests verify boundary handling
- **AC3** `test_mass_scalar_advection_is_conservative_for_constant_velocity` PASSES at 1e-10
- **AC4** Coupled-with-advection scan stable at 60+ steps (currently nonfinite at 30)
- **AC5** 1h coupled probe PASS: sanitize <5%, mu in physical bounds, theta away from clip
- **AC6** 24h coupled probe PASS (this is the M6.x AC5)
- **AC7** Speedup ≥4× (M6.x AC6)
- **AC8** All M4 + M6.x test suites green
- **AC9** Perturbation-pressure advection in c1 explicitly accepted/rejected with WRF citation
- **AC10** ADR-007 → PASS-with-evidence after AC4/AC5/AC6

## Files Worker May Modify

- `src/gpuwrf/dynamics/advection.py` (FIX #1: per-layer dz; FIX #3: mass-conservative form)
- `src/gpuwrf/coupling/physics_couplers.py` (FIX #2: non-periodic interpolation)
- `src/gpuwrf/coupling/driver.py` (only if coupling-order needs adjustment)
- `tests/test_m6x_fallback_c1_*.py` (extend with advection-coupling tests)
- `tests/test_m4_advection.py` (only to confirm existing tests pass)
- `scripts/m6_full_domain_batching.py`
- `artifacts/m6x-fallback-c1/*.json` (regenerate)
- `.agent/decisions/ADR-007-precision-policy.md` (Status amendment after AC4-AC6 green)
- `.agent/decisions/ADR-018-m6x-fallback-c1-tridiag-backend.md` and `ADR-019-m6x-fallback-c1-klemp-skamarock-clean-room.md` (extend with c1-A2 findings)

## Files Worker Must NOT Modify

- `src/gpuwrf/dynamics/acoustic.py` (c1 acoustic core — FROZEN; it works)
- `src/gpuwrf/dynamics/tridiag.py` (FROZEN; verified)
- `src/gpuwrf/dynamics/rk3.py` (FROZEN; mu plumbing works)
- `src/gpuwrf/contracts/state.py` (FROZEN; c1 added base-state fields)
- `src/gpuwrf/physics/**` (frozen per M6.x rule)
- `src/gpuwrf/io/**`, `src/gpuwrf/validation/**` body

## Dispatch

- Worker: codex gpt-5.5 xhigh
- Reviewer: Claude Opus 4.7 xhigh (MANDATORY)
- Wall-time: **4-8h** (specific operators, specific bug-hunt #2 hypotheses, c1 acoustic frozen)
- Worktree: `/tmp/wrf_gpu2_c1` (REUSE — c1-A1 acoustic core is in place)
- Branch: `worker/codex/m6x-c1-klemp-skamarock` (continue same branch)

## HARD RULES

1. NO acoustic-core modifications (frozen — it works)
2. NO new heuristic damping factors (cite WRF for any constant)
3. NO physics-kernel changes
4. Cite WRF source for advection mass-conservation form
5. Cite bug-hunt #2 §2 file:line for the fixes
6. BEFORE `/exit`: `git add . && git commit && git push`
7. `/exit` slash-command

## Decision logic

If 1h coupled probe PASS after c1-A2:
- Continue to 6h + 24h proof
- If 24h PASS + sanitize <5% → M6.x closes GREEN
- Speedup re-run

If 1h coupled probe FAIL after c1-A2:
- Escalate to user (c2 semi-implicit consideration)
- Manager evaluates if perturbation-pressure-advection is the wrong choice

## End-goal context

c1 acoustic is correct. If the advection/coupling fixes land, c1 lands GREEN within hours, not days. The 5-9 day c1 estimate is now ~6-12h total wall time across A1+A2 — much faster than expected.
