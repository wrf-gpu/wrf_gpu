# Sprint Contract — M6 RK Save-Family + Mu Denominator Fix

## Objective

After 2 root-cause fixes today (HPG mass-coupling + acoustic theta mass-coupling), guard-disabled 1h Canary now fails at:

- **theta** explosion at step 47, cell `[11, 31, 67]`, value -226,584,368 K
- Triggered by step 46 diagnostic: **`c1h * muts + c2h = 2.7745`** at that cell (near-singular!)
- First operator: `acoustic`

The previous acoustic-theta-fix worker traced this to "RK/acoustic save-family semantics around `t_1` / `t_save`". The mass denominator goes to ~0, theta projection blows up.

This sprint:
1. Investigate `c1h*muts + c2h` near-singularity (why is muts approaching `-c2h/c1h`?).
2. Investigate `t_save` / `t_1` semantics in RK loop — are we saving the right snapshot?
3. Fix algebra (per WRF reference) OR clamp/regularize the denominator if WRF does so.

## Non-Goals

- NO touching HPG (just fixed).
- NO touching acoustic theta mass-coupling (just fixed).
- NO retuning bounds.
- NO remote push.

## File Ownership

Worktree `/tmp/wrf_gpu2_rksave` on branch `worker/gpt/m6-rk-save-family-fix`.
FIRST: `cd /tmp/wrf_gpu2_rksave`.

Write-only:
- `src/gpuwrf/dynamics/core/acoustic.py` (acoustic_substep_core composition + theta projection)
- `src/gpuwrf/dynamics/mu_t_advance.py` (advance_mu_t_wrf muts update)
- `src/gpuwrf/dynamics/acoustic_wrf.py` (vertical_acoustic_update, mu_continuity_increment if needed)
- `src/gpuwrf/runtime/operational_mode.py` (RK loop save-family — `_with_save_family`)
- `tests/test_m6_rk_save_family_fix.py` (NEW)
- `.agent/sprints/2026-05-26-m6-rk-save-family-fix/` — proofs + worker-report

Read-only:
- WRF Fortran: `module_small_step_em.F` (advance_mu_t, advance_w, small_step_prep, small_step_finish)
- WRF Fortran: `module_em.F` (rk3 loop, save semantics)
- All other source files

## Inputs

1. This contract.
2. `.agent/sprints/2026-05-26-m6-acoustic-theta-fix/worker-report.md` — the acoustic-theta fix that exposed this layer.
3. `.agent/sprints/2026-05-26-m6-acoustic-theta-fix/proof_step46_probe.json` — step 46 diagnostic with c1h*muts+c2h = 2.7745.
4. `.agent/sprints/2026-05-26-m6-acoustic-theta-fix/fixed/proof_first_explosive_step.json` — step 47 failure.

## Hypothesis Space

1. **muts accumulation error in advance_mu_t_wrf** — `muts_new = mut + mu_new`; if mu_new accumulates wrong direction, muts shrinks toward -c2h/c1h.

2. **t_save not refreshed between RK stages** — if t_save is captured at start of RK1 and never updated, RK2/RK3 use stale t_save while mu drifts.

3. **mu_continuity_increment sign error** — `dmu = mu_continuity_increment(next_state, ...)` then `mu_new = mu + dmu`. Sign on dmu may be wrong → mu drifts toward zero.

4. **`fnp` / `fnm` face weights wrong on top boundary** — cell [11,31,67] is k=11, j=31, i=67 — k=11 is interior, j=31, i=67 are interior. So probably not boundary.

5. **`mu_perturbation` not synced with `mu_total`** — if `mu_total` updates but `mu_perturbation = mu_total - mu_base` is computed from stale mu_base, perturbation drifts and propagates back wrong.

6. **`epssm` off-centering parameter sign** — `muave = 0.5 * ((1+epssm) * mu_new + (1-epssm) * mu_old)`; if epssm wrong, muave biased.

## Acceptance Criteria

### Stage 1 — Reproduce + extract bad cell

Run `taskset -c 0-3 python scripts/m6_guard_disabled_debug.py --run-id 20260521_18z_l3_24h_20260522T072630Z --n-steps 50 --output .agent/sprints/2026-05-26-m6-rk-save-family-fix/baseline/`. Confirm step 47 failure. Extract state at step 46 cell [11,31,67] including muts, mu_perturbation, t_save, t_1.

Write `proof_baseline_reproduces.json` + `step46_input_state.npz`.

### Stage 2 — Trace muts trajectory + identify denominator collapse

Run a per-step probe of muts at [11,31,67] from step 0 to step 47. Plot muts vs step. Identify whether muts drifts monotonically or oscillates. Compare to WRF reference (from wrfout, even though hourly).

Write `proof_muts_trajectory.json`.

### Stage 3 — WRF Fortran cross-check

Identify the matching WRF Fortran formula for muts evolution in the RK loop. Compare formula. Cite file:line.

Write `proof_wrf_fortran_crosscheck.json`.

### Stage 4 — Implement + validate

ALL of:
- Guard-disabled 1h Canary 20260521: bounds pass all 360 steps OR push next failure to step >= 100.
- B6 PRESERVED at 0.0 bitwise.
- Multi-step CPU parity 2/10 PRESERVED at 0.0 bitwise.
- 12/12 guard-disabled tests still pass.

Write `proof_fix_validation.json`.

### Stage 5 — Regression test

`tests/test_m6_rk_save_family_fix.py`: load step-46 state → run one acoustic substep → assert c1h*muts+c2h > 50000 Pa at cell [11,31,67] (sea-level surface mu is ~96000 Pa).

### Stage 6 — Worker report

`worker-report.md` with `Summary:`, fix description, named hypothesis matched, proofs, risks, handoff. >=400 bytes.

## Validation Commands

```bash
cd /tmp/wrf_gpu2_rksave
export OMP_NUM_THREADS=4
export PYTHONPATH="src"

# Baseline
taskset -c 0-3 python scripts/m6_guard_disabled_debug.py --run-id 20260521_18z_l3_24h_20260522T072630Z --n-steps 50 --output .agent/sprints/2026-05-26-m6-rk-save-family-fix/baseline/

# (Apply fix)

# Validation
taskset -c 0-3 python scripts/m6_guard_disabled_debug.py --run-id 20260521_18z_l3_24h_20260522T072630Z --n-steps 360 --output .agent/sprints/2026-05-26-m6-rk-save-family-fix/fixed/
taskset -c 0-3 python scripts/m6b6_coupled_step_compare.py --tier all
taskset -c 0-3 python scripts/m6b_real_ic_operational_compare.py --steps 10
taskset -c 0-3 pytest tests/test_m6_guard_disabled_debug.py -v
taskset -c 0-3 pytest tests/test_m6_rk_save_family_fix.py -v

git add -A && git commit -m "[rk save-family fix] $(date -u +%FT%TZ)"
```
