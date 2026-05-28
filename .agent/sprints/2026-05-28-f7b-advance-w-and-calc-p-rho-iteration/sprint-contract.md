# Sprint Contract — F7.B: `advance_w_wrf` full RHS + `calc_p_rho(step=iteration)` (second scoped rewrite)

**Sprint ID**: `2026-05-28-f7b-advance-w-and-calc-p-rho-iteration`
**Worker**: codex gpt-5.5 xhigh
**Branch**: `worker/gpt/f7b-advance-w-and-calc-p-rho-iteration`
**Worktree**: `/tmp/wrf_gpu2_f7b`
**Wall-time**: 3-5 days (per F7-critic recalibration)
**GPU usage**: YES
**Dispatch**: bash-wrapper pattern — codex sees only the work prompt, wrapper handles tmux notify on exit

## Why this sprint

F7.A landed: cross-RK `_1` family carry, `advance_uv_wrf`, loop-entry `calc_p_rho(step=0)`. Result: first critical violation moved from step 1 / RK1 / substep 1 → step 1 / RK3 / substep 8. Acoustic u/v are active but **unstable** (`3.873e+121` magnitude) because `advance_w_wrf` is still a stub and `calc_p_rho(step=iteration)` per substep is missing. F7.A worker explicitly recommends F7.B as the next required repair.

This sprint replaces the stubbed `_diagnose_pressure` + `_advance_geopotential` + `_ph_tend_increment` with WRF-faithful `advance_w_wrf` (full RHS, divergence damping, geopotential update) AND extends `calc_p_rho` to be called per acoustic substep.

## Binding goal (universal)

A JAX-native GPU port of WRF v4 delivering Canary L2/L3 24-72h RMSE on T2/U10/V10 **statistically equivalent** to CPU WRF v4 under **TOST** at predeclared margins on ≥30-case seasonal ensemble; ≥10× speedup preserved.

## Required inputs (read in order)

1. `proofs/f5/wrf_cadence_spec.md` — items 6 (`calc_p_rho` cadence), 7 (`calc_coef_w`), 10 (`advance_w` with full RHS) — binding spec
2. `.agent/sprints/2026-05-28-f7a-save-family-and-advance-uv/worker-report.md` — what F7.A landed, what's still broken
3. `proofs/f7a/audit_summary.md` — current failure pattern (step 1 / RK3 / substep 8)
4. `proofs/f7a/invariant_violations.json` — exact violations after F7.A
5. `proofs/f7a/regression_diff.md` — before/after F7.A diff
6. `.agent/sprints/2026-05-28-f7-critic/critique.md` — F7-critic methodology lessons for AC design
7. WRF Fortran at `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/dyn_em/`:
   - `module_small_step_em.F` `advance_w` lines 1178-1584 — full RHS, Thomas backsub, geopotential update
   - `module_small_step_em.F` `calc_p_rho` lines 492-563 — both `step=0` (already in F7.A) and `step=iteration` mode
   - `module_small_step_em.F` `calc_coef_w` lines 608-649 — c2a / inverse density / acoustic coefficient assembly
   - `solve_em.F` lines 3837-3898 — advance_w call site context (what RHS terms it receives)
   - `solve_em.F` lines 2676-2716 — calc_coef_w call site context
8. Current JAX:
   - `src/gpuwrf/dynamics/core/acoustic.py` — `_advance_geopotential`, `_ph_tend_increment`, `_diagnose_pressure` (the stubs being replaced)
   - `src/gpuwrf/dynamics/core/calc_p_rho.py` — F7.A's step=0 implementation
   - `src/gpuwrf/dynamics/core/small_step_prep.py` — F7.A's `SmallStepPrepState` (extend if needed for `c2a` carry)
   - `src/gpuwrf/dynamics/tridiag_solve.py` — existing Thomas solver to wrap
   - `src/gpuwrf/runtime/operational_mode.py` — `_acoustic_scan`, `acoustic_substep_core` wrapper

## Approach (3 phases)

### Phase 1 — `calc_p_rho(step=iteration)` per-substep extension (1-1.5 days)

Extend `src/gpuwrf/dynamics/core/calc_p_rho.py` (already exists, has step=0):
- Add `calc_p_rho_wrf_iteration(prep, sub_state)` matching WRF `module_small_step_em.F:492-563` step=iteration branch
- Compute `p_pert`, `alpha_pert`, `c2a` updates after each acoustic substep
- The function must receive the post-substep state and update pressure-memory carry for the next substep

In `acoustic_substep_core`:
- After `advance_w_wrf` (Phase 2), call `calc_p_rho_wrf_iteration` to update p/al/c2a for the next substep's `advance_uv` and `advance_mu_t` inputs.

### Phase 2 — `advance_w_wrf` (1.5-2.5 days)

Implement in `src/gpuwrf/dynamics/core/advance_w.py` (NEW file), mirroring WRF `module_small_step_em.F:1178-1584`:

- **RHS assembly**: include large-step `rw_tend`, vertical PGF perturbation, buoyancy from theta + qv, divergence damping coupling via `c2a`, terrain lower boundary contribution
- **Thomas solve**: wrap existing `src/gpuwrf/dynamics/tridiag_solve.py`, but with the WRF acoustic-`c2a` coefficient matrix (NOT the default-ones matrix that current code uses)
- **Geopotential update**: `ph` advances via `ph_tend` from the implicit-w solve — replace the stub `_ph_tend_increment`
- **Sign conventions**: WRF reference for sign of `mu_tend` vs `mu_save` vs `mu_perturbation` — document at each step

Modify `acoustic_substep_core`:
- After `advance_uv_wrf` and `advance_mu_t_core` (both already in F7.A), call new `advance_w_wrf` before `calc_p_rho_wrf_iteration`
- Keep the WRF substep order: `advance_uv → advance_mu_t → advance_w → calc_p_rho(step=iteration)` per `solve_em.F:3088, 3398, 3837, 4164`

### Phase 3 — Verification (0.5-1 day)

a. Re-run F6 audit: `taskset -c 0-3 python scripts/f6_transaction_audit.py --steps 12 --output-dir proofs/f7b`
b. **MUST-PASS gates per F7-critic AC4 discipline**:
   - F6 combination (a) first critical: MUST move past step 1 / RK3 / substep 8 to step ≥2 or LATER substeps within step 1
   - Acoustic u/v max delta MUST drop from F7.A's 3.873e+121 to physically reasonable range (< 10 m/s for first 12 steps with no large-step tendencies)
   - Pressure bound violation MUST not fire at step 2/RK1 (was first violation site in F7.A)
   - `muts_mut_work_mu_consistency` MUST stay clear
c. Re-run 3 F6 unit tests — all must still pass
d. Run 24h forecast pipeline as smoke: `taskset -c 0-3 python scripts/m7_daily_pipeline.py --hours 1` — at least 1 hour completion expected if not blocked

## Acceptance

- **AC1**: `src/gpuwrf/dynamics/core/advance_w.py` exists with WRF file:line references in docstrings.
- **AC2**: `src/gpuwrf/dynamics/core/calc_p_rho.py` has `calc_p_rho_wrf_iteration` function and is called after `advance_w_wrf` in `acoustic_substep_core`.
- **AC3**: `acoustic_substep_core` substep order is `advance_uv → advance_mu_t → advance_w → calc_p_rho_iter`.
- **AC4** (hardened per F7-critic): F6 re-run on combination (a) shows:
  - first critical violation later than step 1 / RK3 / substep 8 (i.e., step 1 / RK3 / substep ≥ 9 or step ≥ 2)
  - acoustic u/v max delta in [0, 10] m/s range (NOT exponential blow-up)
- **AC5**: 3 F6 unit tests still pass.
- **AC6**: `proofs/f7b/regression_diff.md` documents before/after.
- **AC7**: `proofs/f7b/speedup_estimate.json` — current speedup numbers.
- **AC8**: `proofs/f7b/24h_pipeline_smoke.json` — 1+ hour pipeline survival (or honest reason if blocked).
- **AC9**: `worker-report.md` with verdict `F7B_COMPLETE` or `F7B_PARTIAL` + explicit gaps.

## Hard rules

1. **CPU pinning**: `taskset -c 0-3`.
2. **GPU usage**: YES.
3. **Files writable**: `src/gpuwrf/dynamics/core/advance_w.py` (NEW), `src/gpuwrf/dynamics/core/calc_p_rho.py` (extend), `src/gpuwrf/dynamics/core/acoustic.py` (substep_core order + stub replacement), `src/gpuwrf/dynamics/core/small_step_prep.py` (extend if c2a carry needs more state), `src/gpuwrf/dynamics/tridiag_solve.py` (extend Thomas with c2a coefficient matrix if needed), `src/gpuwrf/runtime/operational_mode.py` (minor — `_acoustic_scan` wiring only), `proofs/f7b/**`, `.agent/sprints/2026-05-28-f7b-.../**`.
4. **Files NOT writable**: physics couplers, BC code, comparator scripts, state contracts, governance, plan, ADRs, dynamics/advection.py (F7.C territory).
5. **No remote push.**
6. **Manager repo ONLY**.
7. **DO NOT include tmux send-keys in your work plan** — manager's bash wrapper handles notify on exit.
8. **End with verdict**: `F7B_COMPLETE` / `F7B_PARTIAL`.

## Out of scope (deferred to F7.C+)

- `rk_addtend_dry` (F7.C)
- WRF flux-form mass-coupled advection (F7.C)
- Scalar flux accumulation `sumflux` (F7.D)
- WRF cadence boundary hooks (F7.D)
- XLA fusion / performance optimization (F8)
