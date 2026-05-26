# Sprint Contract — M6 Acoustic Theta Operator Fix (next layer)

## Objective

The HPG fix sprint (`.agent/sprints/2026-05-26-m6-horizontal-pressure-gradient-fix/`) closed the pressure positive-feedback root cause. With HPG fixed + V-suppress workaround removed, the guard-disabled 360-step replay now first fails at:

- **theta**, step 18, cell `[12, 30, 62]`, value **16207 K** (envelope 700 K → ratio 23.1×)
- First operator: **`acoustic`**

This is the next layer. With `disable_guards=True`, theta evolution is exposed. The `acoustic` substep operator (in `dynamics/core/acoustic.py::acoustic_substep_core` + `dynamics/mu_t_advance.py::advance_mu_t_wrf` + `dynamics/acoustic_wrf.py::vertical_acoustic_update`) is producing 23× envelope theta.

The earlier coftz fix and op-theta-fix both diagnosed theta_face source mismatch but neither addressed the actual operational acoustic substep. Now with HPG correct, find and fix the actual theta-tendency bug.

## Non-Goals

- NO modification to HPG. Don't undo the fix.
- NO modification to disable_guards lane (keep diagnostic).
- NO modification to dynamics/core/ unless you can prove the bug lives there (start with mu_t_advance.py + acoustic_wrf.py vertical_acoustic_update).
- NO retuning bounds.
- NO remote push.

## File Ownership

Worktree at `/tmp/wrf_gpu2_acthetafix` on branch `worker/gpt/m6-acoustic-theta-fix`.
FIRST: `cd /tmp/wrf_gpu2_acthetafix`.

Write-only:
- `src/gpuwrf/dynamics/mu_t_advance.py` (advance_mu_t_wrf theta update)
- `src/gpuwrf/dynamics/acoustic_wrf.py` (vertical_acoustic_update — lines ~841-879)
- `src/gpuwrf/dynamics/core/acoustic.py` (acoustic_substep_core composition — modify only if root cause demonstrably lives there)
- `tests/test_m6_acoustic_theta_fix.py` (NEW)
- `.agent/sprints/2026-05-26-m6-acoustic-theta-fix/` — proofs + worker-report

Read-only:
- WRF Fortran reference: `external/wrf/dyn_em/module_small_step_em.F` (advance_mu_t + advance_w + theta update in acoustic loop)
- `.agent/sprints/2026-05-26-m6-horizontal-pressure-gradient-fix/` (the recent fix to understand the pattern)
- `.agent/sprints/2026-05-26-m6b-operational-theta-fix/` (the partial-fix attempt — see what worked + what didn't)

## Inputs

1. This contract.
2. `.agent/sprints/2026-05-26-m6-horizontal-pressure-gradient-fix/worker-report.md` — the latest diagnostic step.
3. `.agent/sprints/2026-05-26-m6-horizontal-pressure-gradient-fix/fixed/proof_first_explosive_step.json` — the named theta@step18 finding.
4. WRF Fortran `module_small_step_em.F` around `advance_mu_t` (line ~1066) and `advance_w` family.

## Hypothesis Space

Apply the same diagnostic pattern as HPG fix:

1. **theta tendency is mass-coupled vs velocity-coupled mismatch** — WRF stores theta×mu (mass-coupled prognostic) while our State.theta is potential temperature. If we update mass-coupled theta and project back to potential temperature, division by changing mu can amplify errors.

2. **`theta_1` reference state stale** — in advance_mu_t_wrf line 133-148, `theta_1` is used for flux; if it's the same array as `theta` and updated between substeps, theta1 drifts.

3. **`fnm`, `fnp` face weights wrong** — these weight `theta_1[k]` and `theta_1[k-1]` to get face theta; off-by-one or sign error.

4. **`ww` (omega) sign error in wdtn** — `wdtn[k] = ww[k] * face_theta`; if ww has wrong sign convention, theta gradient inverted.

5. **`t_2ave` running-average update wrong** — if t_2ave isn't truly running-average but rather instantaneous, theta accumulates.

6. **Missing `msfty` map factor on horizontal theta flux** — line 146 uses msftx; should it be msftx*msfty or just msftx?

Document in `hypothesis_notes.md`.

## Acceptance Criteria

### Stage 1 — Reproduce + extract bad cell

Run `taskset -c 0-3 python scripts/m6_guard_disabled_debug.py --run-id 20260521_18z_l3_24h_20260522T072630Z --n-steps 25 --output .agent/sprints/2026-05-26-m6-acoustic-theta-fix/baseline/`. Confirm step 18 theta=16207 K at cell [12,30,62]. Extract input state at step 17 → cell [12,30,62] for unit-testing.

Write `proof_baseline_reproduces.json` + `step17_input_state.npz` (under `data/fixtures/m6-acoustic-theta-fix/`).

### Stage 2 — WRF Fortran cross-check

Compute by hand (or from WRF Fortran formula) the expected theta tendency at the bad cell. Compare to JAX output. Identify algebraic discrepancy.

Write `proof_wrf_fortran_crosscheck.json`.

### Stage 3 — Implement + validate

ALL of:
- Guard-disabled 1h Canary 20260521: theta in [200,700]K all 360 steps (the contract's blocker test).
- B6 PRESERVED at 0.0 bitwise.
- Multi-step CPU parity 2/10 PRESERVED at 0.0 bitwise.
- 12/12 guard-disabled tests still pass.

Write `proof_fix_validation.json`.

### Stage 4 — Regression test

`tests/test_m6_acoustic_theta_fix.py`: load step-17 input → run one acoustic substep → assert dθ < 50 K/substep at cell [12,30,62].

### Stage 5 — Worker report

`worker-report.md` with `Summary:`, algebraic fix, hypothesis matched, proofs, risks, handoff. >=400 bytes. Note: if you cannot satisfy Stage 3 fully, write `Summary: BLOCKED — <reason>` + name next operator that would explode.

## Validation Commands

```bash
cd /tmp/wrf_gpu2_acthetafix
export OMP_NUM_THREADS=4
export PYTHONPATH="src"

# Stage 1 baseline
taskset -c 0-3 python scripts/m6_guard_disabled_debug.py --run-id 20260521_18z_l3_24h_20260522T072630Z --n-steps 25 --output .agent/sprints/2026-05-26-m6-acoustic-theta-fix/baseline/

# (Apply fix)

# Stage 3 validation
taskset -c 0-3 python scripts/m6_guard_disabled_debug.py --run-id 20260521_18z_l3_24h_20260522T072630Z --n-steps 360 --output .agent/sprints/2026-05-26-m6-acoustic-theta-fix/fixed/
taskset -c 0-3 python scripts/m6b6_coupled_step_compare.py --tier all
taskset -c 0-3 python scripts/m6b_real_ic_operational_compare.py --steps 10
taskset -c 0-3 pytest tests/test_m6_guard_disabled_debug.py -v
taskset -c 0-3 pytest tests/test_m6_acoustic_theta_fix.py -v

git add -A && git commit -m "[acoustic theta fix] $(date -u +%FT%TZ)"
```
