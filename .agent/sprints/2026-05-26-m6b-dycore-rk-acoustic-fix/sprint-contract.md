# Sprint Contract — M6b dycore_rk_acoustic V Tendency Fix

## Objective

V3-521 localization (`.agent/sprints/2026-05-25-m6b-v3-localize-20260521-bound/`) named **`dycore_rk_acoustic`** as the operator producing the 103.7 m/s V violation at step 46 on Gen2 ID `20260521_18z_l3_24h_20260522T072630Z`. The diagnostic budget at the bad cell shows:

- V start (step 46): 82.24 m/s (already pathological)
- After `rk_acoustic` term: 104.22 m/s (delta **+21.98 m/s in one substep** ≈ +2.2 m/s² for the dt step)
- WRF Fortran reference V at the same (lat, lon, level): **4.27 m/s**
- WRF Fortran nearby max V at same level: 4.31 m/s
- WRF Fortran domain max V at the hour: 11.40 m/s

So our JAX dycore is producing ~20× faster V than WRF for this cell. First detectable divergence is step 40; growth is exponential.

This sprint **fixes the V tendency in the acoustic loop** so that the 1h Canary on 20260521 passes physical bounds.

## Non-Goals

- NO modification to `dynamics/core/` if avoidable (try `dynamics/acoustic_wrf.py` or `operational_mode.py` first).
- NO modification to mass/MU continuity unless the V-tendency fix demonstrably needs it.
- NO new validation tier.
- NO retuning of bound thresholds.
- NO remote push.

## File Ownership

Worktree **already created** at `/tmp/wrf_gpu2_acoustic_fix` on branch `worker/gpt/m6b-dycore-rk-acoustic-fix`.
Your FIRST command: `cd /tmp/wrf_gpu2_acoustic_fix`.

Write-only:
- `src/gpuwrf/dynamics/acoustic_wrf.py` (PRIMARY) — likely fix lives in `horizontal_pressure_gradient` (line 336), `acoustic_substep_carry` (945), or `vertical_acoustic_update`.
- `src/gpuwrf/dynamics/acoustic_loop.py` (if needed — the loop driver).
- `src/gpuwrf/runtime/operational_mode.py` (only if needed — composition wrapper).
- `tests/test_m6b_dycore_rk_acoustic_fix.py` (NEW) — regression test.
- `.agent/sprints/2026-05-26-m6b-dycore-rk-acoustic-fix/` — proofs + worker-report.md.

Read-only:
- `src/gpuwrf/dynamics/core/` (locked — only modify if you can prove the V tendency bug lives there)
- All other source files
- `.agent/sprints/2026-05-25-m6b-v3-localize-20260521-bound/` — the localization evidence you're acting on

## Inputs

1. This sprint contract.
2. `.agent/sprints/2026-05-25-m6b-v3-localize-20260521-bound/proof_operator_decomposition.json` — names the suspect.
3. `.agent/sprints/2026-05-25-m6b-v3-localize-20260521-bound/proof_step46_violation.json` — bad cell coordinates + field snapshots at steps 45 & 46.
4. `.agent/sprints/2026-05-25-m6b-v3-localize-20260521-bound/proof_first_divergent_step.json` — earliest divergence is step 40.
5. WRF Fortran reference: `external/wrf/dyn_em/module_small_step_em.F` — particularly the V advance in the acoustic loop. Cross-check your JAX V tendency against the Fortran formula at the bad cell's stagger.
6. Gen2 wrfout truth: `/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T072630Z/wrfout_d02_*` — has hourly snapshots you can interpolate to verify field values.

## Hypothesis Space (rank order — investigate in this order)

1. **Sign error or missing density coupling** in `horizontal_pressure_gradient`'s dv/dt formula. Most likely — the operator was the most-recently-touched in the reframe.
2. **Missing Rayleigh damping / vertical Coriolis term** specifically on V (compare to dU which appears bounded).
3. **dt_sub vs dt misuse** — the acoustic substep advances V with the full dt instead of dt/n_substeps somewhere.
4. **Stagger mismatch** — V is on a Y-face stagger and the pressure gradient may be reading wrong cells on the boundary or near terrain.
5. **Moisture coupling (cqv) error** — `cqv` is applied to dv but might be using a stale or wrong-shape coupling factor.
6. **Boundary forcing not applied** — boundary cells don't get specified boundary tendencies and the interior pressure gradient blows up.

For each hypothesis you investigate, write a 1-paragraph note in `hypothesis_notes.md`. Don't just try things — document.

## Acceptance Criteria

### Stage 1 — Reproduce the defect locally

Run `python scripts/m6b_v3_localize_521.py --run-id 20260521_18z_l3_24h_20260522T072630Z` and confirm you see V=103.72 m/s at step 46 cell [33,53,36] on the un-patched code. Don't change anything yet; just confirm the diagnostic harness works in this worktree.

Write `proof_baseline_reproduces.json`.

### Stage 2 — Implement and validate the fix

Apply the smallest possible fix that:
- Makes the 1h Canary on 20260521 pass physical bounds (|u|,|v| ≤ 100 m/s, |w| ≤ 50 m/s) for all 360 steps.
- Preserves B6 savepoint parity at 0.0 bitwise (`scripts/m6b6_coupled_step_compare.py`).
- Preserves multi-step CPU parity 2/5/10 = 0.0 bitwise (`scripts/m6b_real_ic_operational_compare.py --steps 2,5,10`).
- All 173 existing tests still pass: `pytest -x`.

Write `proof_fix_validation.json` containing all four results.

### Stage 3 — Regression test

Add `tests/test_m6b_dycore_rk_acoustic_fix.py` that:
- Loads the bad-cell state from `proof_step46_violation.json`.
- Runs one acoustic substep with the fix.
- Asserts dV < 5 m/s/substep at that cell (not the runaway +21.98).

### Stage 4 — Worker report

Write `worker-report.md` with `Summary:`, files changed, commands run + outputs (head/tail), proof object paths, the actual hypothesis that matched + which alternatives were ruled out, risks, and handoff. Must include literal `Summary:` token and be >=400 bytes.

## Validation Commands

```bash
cd /tmp/wrf_gpu2_acoustic_fix
export OMP_NUM_THREADS=4
export PYTHONPATH="src"

# Stage 1 — baseline reproduces
taskset -c 0-3 python scripts/m6b_v3_localize_521.py --run-id 20260521_18z_l3_24h_20260522T072630Z --output .agent/sprints/2026-05-26-m6b-dycore-rk-acoustic-fix/baseline/

# (Implement fix in acoustic_wrf.py / acoustic_loop.py)

# Stage 2 validations
taskset -c 0-3 python scripts/m6b6_coupled_step_compare.py  # B6 must stay 0.0 bitwise
taskset -c 0-3 python scripts/m6b_real_ic_operational_compare.py --steps 2,5,10  # CPU multi-step
taskset -c 0-3 python scripts/m6b_v3_localize_521.py --run-id 20260521_18z_l3_24h_20260522T072630Z --output .agent/sprints/2026-05-26-m6b-dycore-rk-acoustic-fix/fixed/
taskset -c 0-3 pytest -x  # all 173 tests

# Stage 3 regression test
taskset -c 0-3 pytest tests/test_m6b_dycore_rk_acoustic_fix.py -v

git add -A && git commit -m "[dycore_rk_acoustic fix] $(date -u +%FT%TZ)"
```

## Handoff

Worker-report.md with the verdict and proofs. If you cannot find a fix that satisfies all 4 Stage-2 criteria, write `Summary: BLOCKED — <reason>` and document which hypothesis was nearest-fit; manager will dispatch a follow-up sprint.
