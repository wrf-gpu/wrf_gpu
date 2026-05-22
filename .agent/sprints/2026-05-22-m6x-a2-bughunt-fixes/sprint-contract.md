# Sprint Contract — M6.x-A2 Bug-Hunt Fix-Hint Application

**Sprint ID**: `2026-05-22-m6x-a2-bughunt-fixes`
**Created**: 2026-05-22 ~00:55
**Status**: ACTIVE
**Trigger**: M6.x worker A1 honest FAIL self-rejected; parallel Opus bug-hunt identified 3 specific bugs with file:line + WRF citations + minimal verification tests. Bug-hunt recommendation: dispatch fix-hint to worker, NOT invoke c1 fallback (concrete defects citable, not algorithm-broken).
**Parallel with**: nothing critical (M7-S0a closed, M6.5-D1 closed)
**Worktree**: `/tmp/wrf_gpu2_m6x` (reuse — WIP code from A1 is in place; fix-hint applied on top)
**Branch**: `worker/codex/m6x-wrf-canonical-dycore` (continue same branch; A2 is incremental on A1 WIP)

## Objective

Apply bug-hunt's 3-step fix sequence to M6.x WIP code in `/tmp/wrf_gpu2_m6x/`. Re-run 6h direct probe after each fix. If all 3 fixes land + 6h probe passes + 24h probe passes + sanitize <5%, M6.x lands GREEN and ADR-007 can flip to PASS.

## The 3 fixes (from bug-hunt report §5)

### FIX #1 (highest confidence — 5-char edit) — VERY HIGH likelihood

**File:line**: `src/gpuwrf/dynamics/acoustic.py:198`

```python
# CURRENT (worker A1):
ph_next = state.ph + dt_sub * w_next

# REPLACE WITH (add GRAVITY_M_S2 factor):
ph_next = state.ph + dt_sub * GRAVITY_M_S2 * w_next
```

**Why**: `state.ph` is geopotential `[m²/s²]`; `dt_sub * w_next` is `[s]·[m/s] = [m]`. Adding them is dimensionally wrong. PH evolves at 1/g of correct rate → `_layer_thickness_m` stale → every vertical operator drifts.

**WRF citation**: `dyn_em/module_small_step_em.F:1583-1584`

**Re-run 6h direct probe after this fix** to test if it alone is sufficient.

### FIX #2 (asymmetric damping) — HIGH likelihood

**File:line**: `src/gpuwrf/dynamics/acoustic.py:189-196` and `acoustic.py:30`

```python
# CURRENT (worker A1):
PRESSURE_IMPLICIT_RELAXATION = 0.05    # acoustic.py:30 — no WRF citation
...
div = _vertical_implicit_mass_weight(state, dt_sub) * (
    _horizontal_divergence_cgrid(state, grid) + _vertical_divergence_cgrid(state)
)
p_next = state.p - PRESSURE_IMPLICIT_RELAXATION * c2 * dt_sub * div    # ← div masked
p_dynamic = _pressure_perturbation(p_next, state.pb)
u_next = state.u - dt_sub * _mass_to_u_face(alpha) * _grad_x_to_u(p_dynamic, grid)  # ← grad unmasked
v_next = state.v - dt_sub * _mass_to_v_face(alpha) * _grad_y_to_v(p_dynamic, grid)
w_explicit = state.w - dt_sub * _mass_to_w_face(alpha) * _grad_z_to_w(p_dynamic, state)

# REPLACE WITH (remove 0.05 + remove mass-weight mask asymmetry):
div = _horizontal_divergence_cgrid(state, grid) + _vertical_divergence_cgrid(state)  # NO MASK
p_next = state.p - c2 * dt_sub * div    # NO 0.05 factor
p_dynamic = _pressure_perturbation(p_next, state.pb)
u_next = state.u - dt_sub * _mass_to_u_face(alpha) * _grad_x_to_u(p_dynamic, grid)
v_next = state.v - dt_sub * _mass_to_v_face(alpha) * _grad_y_to_v(p_dynamic, grid)
w_explicit = state.w - dt_sub * _mass_to_w_face(alpha) * _grad_z_to_w(p_dynamic, state)
```

**Why**: `PRESSURE_IMPLICIT_RELAXATION = 0.05` is 20× under-damping (no WRF citation). Asymmetric div-mask between p update and (u,v,w) update is non-conservative by construction.

**WRF citation**: `dyn_em/module_small_step_em.F:527-528, 562` (no relaxation factor; divergence damping is symmetric `smdiv` correction, not asymmetric prefactor).

**Also audit `MAX_INVERSE_DENSITY = 0.02`** (acoustic.py:31) — clamps `alpha = R*T/p ≈ 0.83` for normal tropospheric cells → horizontal pressure-gradient force 42× too weak everywhere. Either cite WRF or relax to 5.0 / remove.

**Re-run 6h direct probe after this fix**.

### FIX #3 (mu temporal decoupling — only if #1+#2 not enough)

**Files**: `src/gpuwrf/dynamics/rk3.py:42-62` + `src/gpuwrf/dynamics/tendencies.py:67`

**Why**: WRF canonical (`module_small_step_em.F:1102-1108`) updates mu INSIDE the small-step loop using small-step velocities. Worker A1 updates mu once per RK stage on pre-acoustic velocities. Mismatch accumulates.

**Restructure**: move `mu` update into `forward_backward_acoustic`'s `lax.scan` body using small-step `compute_mu_tendency`. More involved than #1+#2.

**Re-run 6h direct probe after this fix**.

## Acceptance (same as M6.x A1 + bug-hunt verification tests)

- **AC1 FIX #1 applied**: acoustic.py:198 has GRAVITY_M_S2 factor; test `test_ph_evolves_with_g_factor_under_uniform_w` from bug-hunt §2 (add to `tests/test_m6x_dycore_completion.py`) PASSES
- **AC2 FIX #2 applied**: PRESSURE_IMPLICIT_RELAXATION removed; asymmetric mask removed; test `test_acoustic_substep_conserves_column_mass_to_round_off` PASSES
- **AC3 6h direct probe PASS**: <5% sanitize firing; mu in physical bounds (no clip at 1000/120000); theta in [200, 350]K (no clip at 150/550)
- **AC4 24h direct probe PASS**: same bounds; nonfinite=0
- **AC5 unit tests PASS**: all M4 + M6.x tests green
- **AC6 speedup re-run**: m6_full_domain_batching.py + verdict ≥4×
- **AC7 If #1+#2 not enough**: FIX #3 applied per bug-hunt §2 hypothesis #3
- **AC8 ADR-007 → PASS**: amend Status with cited measurements
- **AC9 ADR-015 NEW**: WRF-canonical dycore completion ADR
- **AC10 H2D regression closed**: F-3 from M6-S5

## Files Worker May Modify

- `src/gpuwrf/dynamics/{acoustic.py, tendencies.py, rk3.py, step.py}`
- `src/gpuwrf/dynamics/advection.py` (if mu tendency wiring needs adjustment)
- `src/gpuwrf/contracts/state.py` (only if mu_tendency aux state needed for FIX #3)
- `src/gpuwrf/coupling/driver.py` (sanity check sanitize bounds — should NOT change unless cited)
- `tests/test_m6x_*.py` (add the two new tests from bug-hunt §2)
- `scripts/m6_full_domain_batching.py`
- `artifacts/m6/performance/*.json` (regenerate)
- `.agent/decisions/ADR-007-precision-policy.md` (Status amendment after AC3/AC4 green)
- `.agent/decisions/ADR-015-m6x-wrf-canonical-dycore.md` (NEW after AC3/AC4 green)
- `.agent/sprints/2026-05-22-m6x-a2-bughunt-fixes/worker-report.md` (NEW)

## Files Worker Must NOT Modify

- `src/gpuwrf/physics/**` (frozen)
- `src/gpuwrf/io/**` (frozen)
- `src/gpuwrf/validation/**` body (only re-run)

## Dispatch

- Worker: codex gpt-5.5 xhigh
- Reviewer: Claude Opus 4.7 xhigh (MANDATORY)
- Wall-time: **4-8h** (mostly edits to existing A1 code; bug-hunt provides exact targets)
- Worktree: `/tmp/wrf_gpu2_m6x` (REUSE — has A1 WIP code)
- Branch: `worker/codex/m6x-wrf-canonical-dycore` (continue A1 branch)

## HARD RULES

1. Read bug-hunt report FIRST: `/tmp/wrf_gpu2_main_cp/.agent/sprints/2026-05-22-m6x-parallel-bughunt/bughunt-report.md`
2. Apply fixes in ORDER (#1 → re-test → #2 → re-test → #3 if needed)
3. Run the 2 verification tests from bug-hunt §2 BEFORE 6h probe
4. NO physics-kernel changes (frozen)
5. NO new magic-number stabilizers (cite WRF for any constant)
6. BEFORE `/exit`: `git add . && git commit && git push`
7. `/exit` slash-command

## End-goal context

If FIX #1 alone fixes the drift → 5-character edit closed the critical-path; M6 dispatches to M6-S8 + M7. If #1+#2 → still ~30 LOC changes; M6 closes. If #1+#2+#3 → meaningful refactor but still no architectural pivot; M6 closes. If all 3 + still failing → invoke c1 fallback (5-9 day budget per contingency design).

Per plan critic PC-5: this IS the kill-gate; don't extend without decisive evidence.
