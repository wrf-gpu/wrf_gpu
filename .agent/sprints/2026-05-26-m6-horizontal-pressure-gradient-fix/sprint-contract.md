# Sprint Contract — M6 horizontal_pressure_gradient Root-Cause Fix

## Objective

Guard-disabled debug worker (`.agent/sprints/2026-05-26-m6-guard-disabled-debug-impl/`) named **`horizontal_pressure_gradient`** as the first operator producing > 10× envelope value with guards OFF on 20260521. Specifically:

- Step 49, cell `[32, 52, 40]`
- p_perturbation = -314,176 Pa (envelope 5 kPa → ratio 62.8×)
- Earlier in the substep, `horizontal_pressure_gradient(...) → (du_dt, dv_dt)` was the first operator boundary to produce non-physical output

The prior V-suppress workaround (`_m6b_acoustic_tendencies`) papered over this — it replaced the dycore's V tendency with a base tendency. **This sprint replaces that workaround with a real fix in `horizontal_pressure_gradient`**.

## Non-Goals

- NO modification to `dynamics/core/` if avoidable (your fix should be in `acoustic_wrf.py`).
- NO removing or weakening `disable_guards` flag (it's diagnostic; keep it).
- NO removing the microphysics `_valid_mixing_ratio` / `_finite_or_origin` (they're physical-validity checks, not the bug source).
- NO retuning bounds.
- NO remote push.

## File Ownership

Worktree at `/tmp/wrf_gpu2_hpgfix` on branch `worker/gpt/m6-horizontal-pressure-gradient-fix`.
FIRST: `cd /tmp/wrf_gpu2_hpgfix`.

Write-only:
- `src/gpuwrf/dynamics/acoustic_wrf.py` (PRIMARY — `horizontal_pressure_gradient` at line 336)
- `src/gpuwrf/runtime/operational_mode.py` — REMOVE `_m6b_acoustic_tendencies` workaround at line 218-222 (you have a real fix now)
- `tests/test_m6_horizontal_pressure_gradient_fix.py` (NEW)
- `.agent/sprints/2026-05-26-m6-horizontal-pressure-gradient-fix/` — proofs + worker-report

Read-only:
- `src/gpuwrf/dynamics/core/`
- WRF Fortran reference: `external/wrf/dyn_em/module_small_step_em.F` (especially `advance_uv` and `horizontal_pressure_gradient` formula)
- `.agent/sprints/2026-05-26-m6-guard-disabled-debug-impl/` (the localization evidence)

## Inputs

1. This contract.
2. `.agent/sprints/2026-05-26-m6-guard-disabled-debug-impl/worker-report.md` — names horizontal_pressure_gradient as first operator.
3. `proof_first_explosive_operator.json` — has the per-substep trace.
4. `proof_first_explosive_step.json` — bad cell + value.
5. WRF Fortran `module_small_step_em.F`.

## Hypothesis Space

Examine `horizontal_pressure_gradient` (acoustic_wrf.py:336-419 approximately):

1. **Sign error on dv/dt** — most likely if v on Y-face stagger has wrong sign convention.
2. **Stagger mismatch** — pressure at mass cell, du/dv on face stagger; off-by-one neighbor indexing.
3. **Missing density coupling factor** — WRF includes mu*cqu factor; ours might miss it.
4. **Missing top_lid handling** — the `top_lid` flag conditions some terms; if always True/False wrong path.
5. **`al`, `alt` (inverse-density, total) misuse** — multiply when should divide, or vice versa.
6. **dpn (face pressure) wrong** — `x_face_pressure_dpn` / `y_face_pressure_dpn` (line 276+) may have wrong stencil.
7. **`cqu`, `cqv` moisture coupling factors** — only applied to u/v advance; should they affect dpn too?

For each hypothesis: investigate, document in `hypothesis_notes.md`. **Cite specific WRF Fortran line numbers** for comparison.

## Acceptance Criteria

### Stage 1 — Reproduce + isolate

Run `taskset -c 0-3 python scripts/m6_guard_disabled_debug.py --run-id 20260521_18z_l3_24h_20260522T072630Z --n-steps 50 --output .agent/sprints/2026-05-26-m6-horizontal-pressure-gradient-fix/baseline/` with the V-suppress workaround STILL IN PLACE. Confirm step 49 horizontal_pressure_gradient produces bad value. Pull the input state at step 49 (theta, mu, u, v, p, p_pert) and dump it as a fixture for unit-testing.

Write `proof_baseline_reproduces.json` and `step49_input_state.npz`.

### Stage 2 — WRF Fortran cross-check

For the bad cell, compute by hand (or analytically from WRF Fortran) the expected du/dt and dv/dt from `horizontal_pressure_gradient`. Compare to our JAX output. Identify the algebraic discrepancy.

Write `proof_wrf_fortran_crosscheck.json` with:
- WRF formula (cite file:line)
- Our formula (cite file:line)
- Algebraic delta (which term differs)
- Expected fix

### Stage 3 — Implement and validate

REMOVE `_m6b_acoustic_tendencies` workaround (operational_mode.py:218-222 + line that calls it ~ line 354).
Apply the named fix in `horizontal_pressure_gradient`.

ALL of:
- 1h Canary on 20260521: |u|,|v| ≤ 100 m/s, |w| ≤ 50 m/s, theta in [200,700]K all 360 steps WITH guards DISABLED.
- B6 savepoint parity preserved at 0.0 bitwise.
- 12/12 guard-disabled tests still pass.
- All other tests still pass (modulo pre-existing missing fixture).

Write `proof_fix_validation.json`.

### Stage 4 — Regression test

`tests/test_m6_horizontal_pressure_gradient_fix.py`:
- Load `step49_input_state.npz` from Stage 1.
- Run `horizontal_pressure_gradient` with the fix.
- Assert du/dt and dv/dt at cell [32,52,40] are within physical reason (< 1 m/s²·dt).

### Stage 5 — Worker report

`worker-report.md` with `Summary:`, the algebraic fix description, hypothesis matched, proofs, risks, handoff. >=400 bytes.

If BLOCKED: document the nearest hypothesis + the specific WRF Fortran line that differs from ours.

## Validation Commands

```bash
cd /tmp/wrf_gpu2_hpgfix
export OMP_NUM_THREADS=4
export PYTHONPATH="src"

# Baseline
taskset -c 0-3 python scripts/m6_guard_disabled_debug.py --run-id 20260521_18z_l3_24h_20260522T072630Z --n-steps 50 --output .agent/sprints/2026-05-26-m6-horizontal-pressure-gradient-fix/baseline/

# (Apply fix + remove _m6b_acoustic_tendencies)

# Validation
taskset -c 0-3 python scripts/m6_guard_disabled_debug.py --run-id 20260521_18z_l3_24h_20260522T072630Z --n-steps 360 --output .agent/sprints/2026-05-26-m6-horizontal-pressure-gradient-fix/fixed/
taskset -c 0-3 python scripts/m6b6_coupled_step_compare.py --tier all
taskset -c 0-3 python scripts/m6b_real_ic_operational_compare.py --steps 10
taskset -c 0-3 pytest tests/test_m6_guard_disabled_debug.py -v
taskset -c 0-3 pytest tests/test_m6_horizontal_pressure_gradient_fix.py -v

git add -A && git commit -m "[hpg fix] $(date -u +%FT%TZ)"
```

## Handoff

The named fix delivers M6's deep root cause. If validated, M6 acceptance can be re-attempted.
