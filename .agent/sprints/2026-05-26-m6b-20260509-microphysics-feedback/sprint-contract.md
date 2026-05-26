# Sprint Contract — M6b 20260509 Microphysics-Theta Feedback

## Objective

After operational-theta-fix landed (multi-step CPU parity 2/5/10 = 0.0 bitwise, B6 preserved, V3-521 V matches WRF within 0.1 m/s), **20260509 still busts at step 11**:

- Cell `[k=28, j=59, i=72]`
- theta = **2.44×10¹² K** (bound 700 K)
- **qc = 3.22×10⁷** (cloud water mixing ratio — physically should be < 0.01 kg/kg)
- p_perturbation = -5.4×10⁴ Pa
- u = 52 m/s (high but not absurd)
- v = 12 m/s (OK)
- w = 12 m/s (OK)

The qc = 3.2×10⁷ is the smoking gun — Thompson microphysics is producing explosive cloud water condensation, which releases massive latent heat → theta explodes. The flow:

1. Initial small theta perturbation
2. Vertical motion (small w)
3. Thompson saturation adjustment activates
4. qc condenses unphysically large amount (3.2e7 instead of ~1e-3)
5. Latent heat release: theta jumps by ΔT = qc × L/cp ≈ 3.2e7 × 2500 K (Lv/cp) ≈ 8×10¹⁰ K
6. Positive feedback through Clausius-Clapeyron → next step worse → explosion

V3-509 first divergence step 1, delta = 0.0014 K. The seed grows exponentially.

## Non-Goals

- NO modification to dynamics core that would break multi-step CPU parity or B6.
- NO removing Thompson microphysics.
- NO retuning bounds.
- NO remote push.

## File Ownership

Worktree at `/tmp/wrf_gpu2_micro` on branch `worker/gpt/m6b-20260509-microphysics-feedback`.
FIRST: `cd /tmp/wrf_gpu2_micro`.

Write-only:
- `src/gpuwrf/physics/thompson_column.py` (PRIMARY — Thompson microphysics)
- `src/gpuwrf/runtime/operational_mode.py` (only if the coupling boundary itself is wrong)
- `tests/test_m6b_20260509_microphysics_fix.py` (NEW)
- `.agent/sprints/2026-05-26-m6b-20260509-microphysics-feedback/` — proofs + worker-report

Read-only:
- `src/gpuwrf/dynamics/core/`
- `src/gpuwrf/dynamics/mu_t_advance.py`
- `src/gpuwrf/dynamics/acoustic_wrf.py`
- All other source files

## Inputs

1. This contract.
2. `.agent/sprints/2026-05-26-m6b-operational-theta-fix/v3_509/proof_theta_explosion.json` — the bad cell snapshot (qc=3.2e7, theta=2.4e12 K).
3. `.agent/sprints/2026-05-25-m6b-v3-localize-20260509-theta/proof_first_bad_step_tracer.json` — first divergence step 1, delta 0.0014 K.
4. WRF Fortran reference `external/wrf/phys/module_mp_thompson.F` — the saturation adjustment in Thompson.
5. Gen2 wrfout truth at `/mnt/data/canairy_meteo/runs/wrf_l3/20260509_18z_l3_24h_20260511T190519Z/` — WRF reference qc, qv, theta at that cell.

## Hypothesis Space

1. **Thompson saturation adjustment overshoots** — Newton iteration or linearized correction unbounded.
2. **qv → qc conversion sign error** — too much vapor converted per step.
3. **Operational skips a clamp/limiter** — Thompson in our code may produce sensible qc, but operational doesn't clamp [0, 0.02] kg/kg.
4. **Latent heat coupling formula error** — theta = theta + (Lv/cp) * Δqc with wrong sign or Lv constant.
5. **Initial qc / qv inputs corrupted** — from boundary forcing or wrfinput conversion (check wrfinput_d02 for 20260509 IC and compare to WRF).
6. **Time-step too large for Thompson** — Thompson typically uses dt=10s, but if our operational mode passes a different dt, condensation explodes.

Document each in `hypothesis_notes.md`.

## Acceptance Criteria

### Stage 1 — Reproduce + localize

Run `taskset -c 0-3 python scripts/m6b_v3_localize_509.py --run-id 20260509_18z_l3_24h_20260511T190519Z`. Confirm step 11 theta=2.4e12 K and qc=3.2e7.

Then run `diagnostic_limiter_activation_tracker.py` to check whether any clamp/limiter is firing (or NOT firing when it should). Run `diagnostic_warm_bubble_vs_slice.py` to compare against an idealized warm-bubble test (which should NOT explode).

Write `proof_baseline_reproduces.json`.

### Stage 2 — Compare to WRF reference

Pull WRF qc, qv, theta values from wrfout_d02 at hour 1 for the bad cell vicinity. Confirm WRF reference qc is physical (< 0.01 kg/kg). The delta between our qc and WRF qc at step 11 quantifies the bug.

Write `proof_wrf_microphysics_reference.json`.

### Stage 3 — Fix + validate

ALL of:
- 1h Canary on 20260509: theta stays in [200,700]K all 360 steps + qc < 0.05 kg/kg.
- Multi-step CPU parity preserved at 0.0 bitwise (`scripts/m6b_real_ic_operational_compare.py --steps 2`, 5, 10).
- B6 preserved (`scripts/m6b6_coupled_step_compare.py --tier all`).
- 1h Canary on 20260521 still passes bounds.

Write `proof_fix_validation.json`.

### Stage 4 — Regression test

`tests/test_m6b_20260509_microphysics_fix.py`: load step-N cell with realistic qv/qc/theta, run one Thompson step, assert dqc < 0.005 kg/kg and dθ < 5 K/step.

### Stage 5 — Worker report

`worker-report.md` with `Summary:`, hypothesis matched, fix description, proofs, risks, handoff. >=400 bytes.

## Validation Commands

```bash
cd /tmp/wrf_gpu2_micro
export OMP_NUM_THREADS=4
export PYTHONPATH="src"

# Baseline
taskset -c 0-3 python scripts/m6b_v3_localize_509.py --run-id 20260509_18z_l3_24h_20260511T190519Z --output .agent/sprints/2026-05-26-m6b-20260509-microphysics-feedback/baseline/
taskset -c 0-3 python scripts/diagnostic_limiter_activation_tracker.py --output .agent/sprints/2026-05-26-m6b-20260509-microphysics-feedback/ || true

# (Apply fix)

# Validation
taskset -c 0-3 python scripts/m6b_real_ic_operational_compare.py --steps 2
taskset -c 0-3 python scripts/m6b_real_ic_operational_compare.py --steps 10
taskset -c 0-3 python scripts/m6b6_coupled_step_compare.py --tier all
taskset -c 0-3 python scripts/m6b_v3_localize_509.py --run-id 20260509_18z_l3_24h_20260511T190519Z --output .agent/sprints/2026-05-26-m6b-20260509-microphysics-feedback/fixed/
taskset -c 0-3 python scripts/m6b_v3_localize_521.py --run-id 20260521_18z_l3_24h_20260522T072630Z --output .agent/sprints/2026-05-26-m6b-20260509-microphysics-feedback/fixed_521/
taskset -c 0-3 pytest tests/test_m6b_20260509_microphysics_fix.py -v

git add -A && git commit -m "[micro-feedback fix] $(date -u +%FT%TZ)"
```

## Handoff

Per universal spec.
