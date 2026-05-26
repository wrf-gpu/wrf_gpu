# Sprint Contract — M6b coftz Theta Tendency Fix (V3-509)

## Objective

V3-509 localization (`.agent/sprints/2026-05-25-m6b-v3-localize-20260509-theta/`) named **`coftz`** as the operator producing the theta explosion on Gen2 ID `20260509_18z_l3_24h_20260511T190519Z`. Diagnostic:

- First theta bound breach: **step 11 / lead 110 s** at cell k=28 j=60 i=73 with theta = **2.6×10¹² K** (bound: 400 K).
- WRF Fortran reference at same cell: **348.6 K** (benign — no convective signal).
- First detectable divergence vs WRF: **step 1**, delta = 0.0014 K.
- Boundary forcing: OK (not a forcing problem).
- Top operator: `coftz` (vertical theta-tendency coefficient in the implicit vertical acoustic solver).

This is a SLOW-GROWING instability (0.0014 K at step 1 → 2.6e12 K at step 11) — characteristic of an accumulator/sign error in the implicit theta update, not a single bad cell.

This sprint **fixes the coftz / vertical_implicit theta tendency** so 1h Canary on 20260509 passes physical bounds.

## Non-Goals

- NO modification to `dynamics/core/` if avoidable.
- NO retuning bounds.
- NO new validation tier.
- NO touching `horizontal_pressure_gradient` (that's the parallel acoustic-fix sprint).
- NO remote push.

## File Ownership

Worktree **already created** at `/tmp/wrf_gpu2_coftz_fix` on branch `worker/gpt/m6b-coftz-theta-fix`.
Your FIRST command: `cd /tmp/wrf_gpu2_coftz_fix`.

Write-only:
- `src/gpuwrf/dynamics/vertical_implicit_solver.py` (PRIMARY — coftz construction).
- `src/gpuwrf/dynamics/acoustic_wrf.py` lines 841-879 (the `vertical_acoustic_update` that USES coftz).
- `tests/test_m6b_coftz_theta_fix.py` (NEW).
- `.agent/sprints/2026-05-26-m6b-coftz-theta-fix/` — proofs + worker-report.md.

Read-only:
- `src/gpuwrf/dynamics/core/`.
- `src/gpuwrf/dynamics/acoustic_wrf.py` outside lines 841-879 (this is the parallel acoustic-fix sprint's territory).
- `.agent/sprints/2026-05-25-m6b-v3-localize-20260509-theta/` — the localization evidence.

## Inputs

1. This sprint contract.
2. V3-509 localization proofs (theta explosion + operator budget + WRF reference + first divergent step).
3. WRF Fortran reference: `external/wrf/dyn_em/module_small_step_em.F` — specifically the vertical implicit theta solver coefficients (`small_step_prep` and `advance_w` family).
4. Gen2 wrfout truth: `/mnt/data/canairy_meteo/runs/wrf_l3/20260509_18z_l3_24h_20260511T190519Z/`.

## Hypothesis Space (rank order)

1. **`coftz` sign or factor error** — the coefficient `dtseps * theta_face` (line 53 of vertical_implicit_solver.py) might have wrong sign convention or missing `1/Pres` factor relative to WRF Fortran.
2. **`theta_face` source mismatch** — WRF computes theta_face from `t_2ave` (running average) but our code may use instantaneous theta, causing positive feedback.
3. **`rdzw` stagger error** — `coftz[1:, :] * rw_p[1:, :] - coftz[:-1, :] * rw_p[:-1, :]` divergence formula may be on wrong stagger.
4. **`build_epssm_column_coefficients` time-weighting wrong** — `epssm` is the WRF off-centering parameter for acoustic stability; if it's 0 or out of [0,1], the implicit step is unstable.
5. **t_2ave / theta double-update** — both validation_wrappers and operational might be advancing theta in the acoustic loop AND in advance_mu_t, causing 2× growth per step.
6. **GPU vs CPU theta arithmetic** — though gpu-cpu sprint said CPU/GPU agree exactly, the validation_wrappers theta=9e8 K at step 2 is the same as gpu/cpu — could still be a precision issue.

Document each hypothesis you investigate in `hypothesis_notes.md`.

## Acceptance Criteria

### Stage 1 — Reproduce locally

Run `python scripts/m6b_v3_localize_509.py --run-id 20260509_18z_l3_24h_20260511T190519Z` and confirm step-11 theta=2.6e12 K at cell [28,60,73] on un-patched code.

Write `proof_baseline_reproduces.json`.

### Stage 2 — Implement and validate fix

Smallest possible fix that satisfies ALL:
- 1h Canary on 20260509 passes per-level theta bounds (lower 30: [200,400]K; upper 14: [250,700]K) all 360 steps.
- B6 savepoint parity preserved at 0.0 bitwise (`scripts/m6b6_coupled_step_compare.py`).
- Multi-step CPU parity 2/5/10 = 0.0 bitwise (`scripts/m6b_real_ic_operational_compare.py --steps 2,5,10`).
- All 173 tests pass: `pytest -x`.
- Also check 20260521 still has only the V momentum issue (this fix shouldn't perturb the acoustic-fix sprint's territory).

Write `proof_fix_validation.json`.

### Stage 3 — Regression test

`tests/test_m6b_coftz_theta_fix.py`: load bad-cell state from `proof_theta_explosion.json`, run one vertical acoustic update, assert dθ < 1 K/substep at that cell.

### Stage 4 — Worker report

`worker-report.md` with `Summary:`, fix description, hypothesis that matched + alternatives ruled out, proofs, risks, handoff. >=400 bytes.

## Validation Commands

```bash
cd /tmp/wrf_gpu2_coftz_fix
export OMP_NUM_THREADS=4
export PYTHONPATH="src"

# Stage 1 baseline
taskset -c 0-3 python scripts/m6b_v3_localize_509.py --run-id 20260509_18z_l3_24h_20260511T190519Z --output .agent/sprints/2026-05-26-m6b-coftz-theta-fix/baseline/

# (Apply fix in vertical_implicit_solver.py / acoustic_wrf.py)

# Stage 2 validations
taskset -c 0-3 python scripts/m6b6_coupled_step_compare.py
taskset -c 0-3 python scripts/m6b_real_ic_operational_compare.py --steps 2,5,10
taskset -c 0-3 python scripts/m6b_v3_localize_509.py --run-id 20260509_18z_l3_24h_20260511T190519Z --output .agent/sprints/2026-05-26-m6b-coftz-theta-fix/fixed/
taskset -c 0-3 pytest -x
taskset -c 0-3 pytest tests/test_m6b_coftz_theta_fix.py -v

git add -A && git commit -m "[coftz theta fix] $(date -u +%FT%TZ)"
```

## Handoff

`worker-report.md` with verdict. If BLOCKED, name the nearest-fit hypothesis.
