# Sprint Contract — c1-A3 Long-Time Integration Fix

**Sprint ID**: `2026-05-22-m6x-c1-a3-longtime-fix`
**Created**: 2026-05-22 ~03:10
**Status**: ACTIVE
**Trigger**: c1-A2 fixed 4 operators; 18-step probe clean but 1h probe 88.6% sanitize. Bug-hunt #3 found 3 long-time accumulators (file:line + WRF citations):
- **A** Periodic-wrap `jnp.roll` in dycore stencils (acoustic.py + advection.py) — c1-A2 fixed physics_couplers but missed dynamics
- **B** Inconsistent dz floor (`acoustic.py:_layer_thickness_m` no floor, `advection.py:_dz_from_state` 1m floor) → tridiag explosion when dz collapses
- **C** Missing Klemp §3d divergence damper + missing horizontal ph advection

## The 5-step fix sequence (per bughunt3 §5)

### FIX #A (FIRST — cheapest, biggest leverage)
Replace `jnp.roll` periodic stencils with edge-mirror extrapolation in these 7 dycore call sites:
- `acoustic.py:139-147` `_grad_x_to_u`, `_grad_y_to_v`
- `acoustic.py:162-170` `_mass_to_u_face_2d`, `_mass_to_v_face_2d`
- `advection.py:218-229` `_periodic_flux5_faces`
- `advection.py:303-312` `_mass_to_u_face`, `_mass_to_v_face`

Use the SAME edge-mirror pattern c1-A2 applied at `physics_couplers.py:98-109`.

**Expected after FIX #A**: 1h probe `nonfinite_count` drops ≥10×, `fired_steps` drops ≥5×.

### FIX #B
Add `jnp.where(dz > 0.0, jnp.maximum(dz, 1.0), _flat_dz(grid))` to `acoustic.py:_layer_thickness_m` (1-line change matching `advection.py:_dz_from_state`).

### FIX #C(C1)
Add Klemp 2007 §3d `smdiv` divergence damping in `acoustic_once`:
- Carry `p_prev` in the acoustic `lax.scan` carry
- After main pressure update: `p_next = p_next + smdiv * (p_next - p_prev)`
- Start with `smdiv = 0.1` per WRF default
- WRF citation: `module_small_step_em.F:557-565`

### FIX #C(C2)
Add `ph` to `compute_advection_tendencies` using mass-conservative scalar advection (same pattern as theta/qv/p_perturbation).
- Note: ph lives on w-faces (nz+1) — needs face-collocation pass
- WRF citation: `module_em.F:1292` `advect_ph_implicit`; `module_big_step_utilities_em.F` `rhs_ph`

### Pre-flight DIAGNOSTIC (before FIX #A): sanitize-disable probe
Run 1h coupled probe with sanitize DISABLED. Expected: dycore goes non-finite by step 50 (proves sanitize is symptom-catcher, not driver). If sanitize-on lasts longer than sanitize-off, sanitize is contributing — informative either way.

## Acceptance

- **AC1** FIX #A applied to all 7 named dycore call sites; new tests verify non-periodic behavior at boundaries
- **AC2** FIX #B applied (1-line dz floor); regression test for dz>0 small case
- **AC3** Pre-flight sanitize-disable probe run; result documented
- **AC4** FIX #A 1h probe: `nonfinite_count` < 1/10 of c1-A2 baseline
- **AC5** FIX #C(C1) smdiv damper added with `p_prev` in scan carry
- **AC6** FIX #C(C2) ph horizontal advection added
- **AC7** 1h coupled probe PASS: sanitize <5%, mu in physical bounds, theta away from clip
- **AC8** 24h coupled probe PASS (M6.x AC5)
- **AC9** Speedup ≥4× (M6.x AC6)
- **AC10** ADR-007 → PASS-with-evidence after AC7/AC8/AC9

## Files Worker May Modify

- `src/gpuwrf/dynamics/acoustic.py` (FIX #A stencils + FIX #B dz floor + FIX #C1 smdiv)
- `src/gpuwrf/dynamics/advection.py` (FIX #A stencils + FIX #C2 ph advection)
- `src/gpuwrf/dynamics/rk3.py`, `step.py` (only if scan carry needs adjustment for p_prev or ph)
- `src/gpuwrf/coupling/driver.py` (sanitize-disable flag for diagnostic only)
- `tests/test_m6x_fallback_c1_*.py` (extend per fix)
- `tests/test_m4_advection.py`, `tests/test_m4_acoustic.py` (only to confirm pass)
- `artifacts/m6x-fallback-c1/*.json` (regenerate)
- `.agent/decisions/ADR-007-precision-policy.md` (Status amendment after AC7-AC9 green)
- `.agent/decisions/ADR-018`, `ADR-019` (extend with c1-A3 findings)

## Files Worker Must NOT Modify

- `src/gpuwrf/contracts/state.py` (FROZEN — c1-A1 base-state was correct)
- `src/gpuwrf/dynamics/tridiag.py` (FROZEN — c1-A1 Thomas solve verified)
- `src/gpuwrf/coupling/physics_couplers.py` (c1-A2 fixed; don't re-touch)
- `src/gpuwrf/physics/**` (frozen)
- `src/gpuwrf/io/**`, `src/gpuwrf/validation/**` body

## Dispatch

- Worker: codex gpt-5.5 xhigh
- Reviewer: Claude Opus 4.7 xhigh (MANDATORY)
- Wall-time: **4-8h**
- Worktree: `/tmp/wrf_gpu2_c1` (REUSE)
- Branch: `worker/codex/m6x-c1-klemp-skamarock` (continue)

## HARD RULES

1. Apply fixes in ORDER (Pre-flight → A → B → C1 → C2); re-run 1h probe after each
2. NO new heuristic damping factors beyond smdiv=0.1 (cite WRF)
3. NO physics-kernel changes
4. Cite bughunt3 §3 file:line for each fix
5. Cite WRF source for smdiv (`module_small_step_em.F:557-565`) and ph advection (`module_em.F:1292`)
6. BEFORE `/exit`: `git add . && git commit && git push`
7. `/exit` slash-command

## Decision logic

- If FIX #A alone brings 1h probe to <10% firing → all 3 hypotheses combined likely closes
- If after FIX #A + B + C → 1h still failing → bughunt3 §5 said escalate (re-open bug-hunt #2 Hyp A buoyancy, or c2 territory)

## End-goal

c1 iteration #6 (A1+A2+A3). If this lands: total c1 wall ≈ <8h vs 5-9 day estimate. M6.x closes, M7-S0 + M6-S8 unblock simultaneously.
