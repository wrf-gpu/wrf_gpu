# Sprint Contract — M6b Real-IC Operational Composition Bisection (v2)

## Objective

The previous bisection (commit `worker/gpt/m6b-operational-composition-bisection`) localized "RK1 acoustic loop omission" on a synthetic-IC controlled comparison. The fix landed at `879ef56`. The combined acceptance sprint (commit `worker/gpt/m6b-rk1-d2h-acceptance`) re-tested on **4 real Gen2 wrfout-rich IDs** and found **RK1 parity STILL FAILS at step 1**.

This means either:
1. Synthetic-IC bisection's "RK1 omission" finding was a controlled-test artifact (RK1 fix didn't address the actual real-IC defect)
2. RK1 fix is correct but a SECOND composition defect triggers only on real-IC paths

This sprint runs a bisection AGAIN — but now on **real Gen2 wrfout-rich IC** (`20260521_18z_l3_24h_20260522T072630Z`), with the RK1 fix already in tree. Localize the first diverging operator/stage/field on real data.

## Non-Goals

- NO fix attempt. Diagnosis only.
- NO modifications to validation-mode code.
- NO modifications to operational `wrf.exe`.
- NO synthetic IC. Real Gen2 wrfout only.
- NO sanitizer.
- NO 1h forecast.
- NO multi-step replay (focus on step 1 only; localize at sub-step level).
- NO remote push.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_realbisect` on branch `worker/gpt/m6b-real-ic-bisection`.

Write-only:
- `scripts/m6b_real_ic_operational_compare.py` (NEW) — side-by-side runner on real Gen2 IC, per-RK / per-acoustic-substep / per-operator delta
- `tests/test_m6b_real_ic_bisection.py` (NEW)
- `.agent/sprints/2026-05-25-m6b-real-ic-bisection/` — proofs + worker-report

Read-only:
- `src/gpuwrf/runtime/operational_mode.py` (post RK1+D2H fixes)
- `src/gpuwrf/dynamics/coupled_step.py` + `acoustic_loop.py` + `dycore_step.py` (validation-only — read for ordering)

## Inputs (mandatory)

1. This sprint contract
2. `.agent/sprints/2026-05-25-m6b-operational-composition-bisection/worker-report.md` (the prior synthetic-IC bisection; for comparison)
3. `.agent/sprints/2026-05-25-m6b-rk1-d2h-acceptance/worker-report.md` (the real-IC failure evidence)
4. `.agent/sprints/2026-05-25-m6b-rk1-d2h-acceptance/proof_rk1_parity_step1_*.json` (per-ID per-field deltas — start here)
5. `src/gpuwrf/runtime/operational_mode.py` (current, with RK1 fix at 879ef56)
6. Gen2 IC: `/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T072630Z/` (9 wrfouts; canonical real IC)

## Acceptance Criteria

### Stage 1 — Side-by-side on real Gen2 IC (MANDATORY)

`scripts/m6b_real_ic_operational_compare.py`:
- Load Gen2 IC from `wrfout_d02_2026-05-21_18:00:00` (or first available) — REAL state, not synthetic
- Run operational_mode for 1 timestep (3 RK stages, 10 acoustic substeps each)
- Run validation `coupled_step` from the SAME real IC
- At every RK stage end and every acoustic substep end: per-field max-abs delta

Capture: `proof_real_ic_step1_full_trace.json`.

### Stage 2 — Localize the first diverging stage (MANDATORY)

Find the FIRST stage where any field's delta exceeds 1e-10:
- RK1 advection candidate (pre-acoustic)
- RK1 acoustic substep 1 (the one supposedly added by 879ef56 — VERIFY IT IS BEING CALLED)
- RK1 acoustic substep 2-10
- RK1 post-acoustic
- RK2 advection candidate
- ... etc.

If divergence is at "RK1 advection candidate" (before any acoustic): the defect is in the pre-acoustic advection or in IC reading.
If divergence is at "RK1 acoustic substep 1": the RK1 fix added the call but the call has a defect (wrong inputs, wrong scratch initialization, wrong operator).
If divergence is at "RK2..." or later: the RK1 fix is correct; second defect lives downstream.

Capture: `proof_first_diverging_stage.json`.

### Stage 3 — Per-operator drill-down at the first diverging stage (MANDATORY)

At the first diverging stage, drill into per-operator delta:
- `calc_coef_w` outputs (a, alpha, gamma)
- `advance_uv` outputs (u, v, pressure-gradient terms)
- `advance_mu_t` outputs (mu, mudf, muts, muave, ww, theta, ph_tend)
- Thomas forward sweep (intermediate alpha/gamma if instrumentable)
- Thomas back sub (w)
- Scratch updates (t_2ave, ww, muave running averages)
- Rayleigh damping
- PH final

Name the operator with the largest delta. Cite WRF source `dyn_em/module_small_step_em.F` for the canonical behavior.

Capture: `proof_first_diverging_operator.json`.

### Stage 4 — Verify RK1 fix is actually invoked (MANDATORY)

Add instrumentation to operational_mode.py (validation-only `debug=True` static-arg path; DCE-eliminated in production) that prints a marker each time the RK1 acoustic loop is entered. Re-run; confirm marker fires.

If marker fires: 879ef56 IS called on real IC; the divergence comes from inside the loop or from later code.
If marker doesn't fire: 879ef56 is conditionally gated off in real-IC path; document the gate condition.

Capture: `proof_rk1_fix_invocation.txt`.

### Stage 5 — Verdict memo

`worker-report.md` answers:
1. Is the synthetic-IC bisection's "RK1 omission" finding still valid on real-IC? (YES if RK1 fix is invoked and divergence is downstream; NO if RK1 fix not invoked.)
2. What's the FIRST diverging stage on real-IC?
3. What's the named defect (file:line + WRF source citation)?
4. Recommended next fix sprint (single-line scope).

## Validation Commands

```bash
cd /tmp/wrf_gpu2_realbisect
taskset -c 0-3 python scripts/m6b_real_ic_operational_compare.py --gen2-run-id 20260521_18z_l3_24h_20260522T072630Z --gen2-ic-time "2026-05-21_18:00:00" 2>&1 | tee .agent/sprints/2026-05-25-m6b-real-ic-bisection/proof_bisection_run.txt
pytest tests/test_m6b_real_ic_bisection.py -v 2>&1 | tee .agent/sprints/2026-05-25-m6b-real-ic-bisection/proof_no_regression.txt
```

## Kill Gates

- Cannot find a divergence in 1 timestep → either operational mode is now correct OR the harness has a bug. Document.
- Operational sha changes → STOP.

## Risks

- Loading real Gen2 IC may surface IC-reader bugs (e.g., grid alignment, units, staggering). Document if so.
- The 879ef56 RK1 fix may have a subtle conditional that's true for synthetic IC and false for real — check via Stage 4 instrumentation first.

## Handoff Requirements

When defect localized + memo committed: `/exit`. Manager dispatches narrow fix sprint.

Time budget: **45-90 min**.
