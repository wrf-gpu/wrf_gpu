# Sprint Contract — M6b Operational Composition Fix: RK1 Acoustic Loop Omission

## Objective

Bisection (commit `worker/gpt/m6b-operational-composition-bisection`) localized the operational defect: **`operational_mode.py` skips the acoustic substep loop entirely at RK stage 1**. Validation `coupled_timestep_wrf` runs `acoustic_substep_wrf` at all 3 RK stages. At step 1, theta is already off by 5506K (1.0e300 nonfinite sentinel by step 62).

Fix: add the acoustic substep loop at RK1, matching validation's RK-stage semantics from WRF source.

## Non-Goals

- NO modifications to validation-mode code (locked).
- NO modifications to operational `wrf.exe`.
- NO new clamps, sanitizer, stabilizers.
- NO changes to acoustic_loop.py / coupled_step.py (validation-only, locked).
- NO imports of validation-only helpers into operational_mode.py (Critic Amendment #1 still binding).
- NO d2h-inside-loop fix in this sprint (parallel sprint handles it).
- NO 1h forecast (1-timestep + 10-timestep probes only).
- NO remote push.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_rk1fix` on branch `worker/gpt/m6b-fix-rk1-acoustic-loop`.

Write-only:
- `src/gpuwrf/runtime/operational_mode.py` — add acoustic substep loop at RK1; verify RK2/RK3 are correct
- `tests/test_m6b_fix_rk1_acoustic_loop.py` (NEW)
- `.agent/sprints/2026-05-25-m6b-fix-rk1-acoustic-loop/` — proofs + worker-report

Read-only everywhere else.

## Inputs (mandatory)

1. This sprint contract
2. `.agent/sprints/2026-05-25-m6b-operational-composition-bisection/worker-report.md` (localization + WRF source citation)
3. `.agent/sprints/2026-05-25-m6b-operational-composition-bisection/proof_bisection_substep_level.json` (the per-substep delta evidence)
4. `src/gpuwrf/dynamics/coupled_step.py` (validation reference — DO NOT import; READ to understand correct RK ordering)
5. `src/gpuwrf/dynamics/acoustic_loop.py` (validation reference — same)
6. `src/gpuwrf/runtime/operational_mode.py` (file to fix)
7. WRF source: `dyn_em/solve_em.F` (the RK3 outer loop with acoustic at all 3 stages)

## Acceptance Criteria

### Stage 1 — Apply the fix (MANDATORY)

In `operational_mode.py`: add the acoustic substep loop body at RK stage 1. Match the RK2/RK3 implementation pattern (presumably already correct since bisection identified only RK1). Cite WRF `dyn_em/solve_em.F` for the correct RK-acoustic ordering.

Fix MUST NOT import `acoustic_loop.py` / `coupled_step.py` (validation-only). Copy/adapt with citation, per Amendment #1.

### Stage 2 — 1-step operational-vs-validation parity check (MANDATORY)

Re-run the bisection comparator `scripts/m6b_operational_vs_validation_compare.py --gen2-run-id 20260523_18z_l3_24h_20260524T004313Z --steps 1`. Acceptance: max-abs delta across all fields < 1e-10 (ideally FP64 ULP).

If divergence remains: the fix is incomplete; either RK2/RK3 also need attention or there's a secondary defect. Document.

Capture: `proof_step1_parity.json`.

### Stage 3 — 10-step bisection probe (MANDATORY)

Same comparator, 10 steps. Acceptance: max-abs delta across all fields stays bounded (< 1e-8 due to composition error compounding).

Capture: `proof_step10_parity.json`.

### Stage 4 — 70-step probe (MANDATORY, the original failure point)

Same comparator, 70 steps (covers the original step-62 failure). Acceptance: no nonfinite, bounded theta within per-level bounds (lower 30 levels < 400K).

Capture: `proof_step70_probe.json`.

### Stage 5 — B6 validation regression check (MANDATORY)

`python scripts/m6b6_coupled_step_compare.py --tier golden`. Acceptance: still `SEVENTH-COUPLED-STEP-PARITY-ACHIEVED` with `max_abs_delta: 0.0` — validation mode must NOT regress.

Capture: `proof_b6_regression.txt`.

### Stage 6 — No regression

```bash
pytest tests/test_m6x_*.py tests/test_m3_transfer_audit.py tests/test_m6b0_*.py tests/test_m6b0r_*.py tests/test_m6b1_*.py tests/test_m6b2_*.py tests/test_m6b3_*.py tests/test_m6b_hygiene_*.py tests/test_m6b4_*.py tests/test_m6b5_*.py tests/test_m6b6_*.py tests/test_m6_operational_*.py tests/test_m6_perf_*.py tests/test_m6b_carry_expansion_*.py tests/test_m6b_d2h_warmed_*.py tests/test_m6b_honest_v2_*.py tests/test_m6b_operational_vs_validation_*.py tests/test_m6b_fix_rk1_*.py -v
```

### Stage 7 — Worker report

`worker-report.md`: diff summary (the added acoustic loop body + RK-ordering rationale + WRF citation), Stage 2-4 per-step deltas, B6 regression status, files changed, **M6b RETRY V3 dispatch recommendation**.

## Validation Commands

```bash
cd /tmp/wrf_gpu2_rk1fix
taskset -c 0-3 python scripts/m6b_operational_vs_validation_compare.py --gen2-run-id 20260523_18z_l3_24h_20260524T004313Z --steps 1 2>&1 | tee .agent/sprints/2026-05-25-m6b-fix-rk1-acoustic-loop/proof_step1_parity.txt
taskset -c 0-3 python scripts/m6b_operational_vs_validation_compare.py --gen2-run-id 20260523_18z_l3_24h_20260524T004313Z --steps 10 2>&1 | tee .agent/sprints/2026-05-25-m6b-fix-rk1-acoustic-loop/proof_step10_parity.txt
taskset -c 0-3 python scripts/m6b_carry_expansion_probe.py --runs 1 --duration-s 70 2>&1 | tee .agent/sprints/2026-05-25-m6b-fix-rk1-acoustic-loop/proof_step70_probe.txt
taskset -c 0-3 python scripts/m6b6_coupled_step_compare.py --tier golden 2>&1 | tee .agent/sprints/2026-05-25-m6b-fix-rk1-acoustic-loop/proof_b6_regression.txt
pytest <full test list> -v 2>&1 | tee .agent/sprints/2026-05-25-m6b-fix-rk1-acoustic-loop/proof_no_regression.txt
```

## Kill Gates

- Step-1 parity not within 1e-10 → fix is incomplete; document remaining divergence; route to follow-up sprint.
- B6 regression → REJECT (validation must stay 0.0).
- Imports validation-only helpers → REJECT.
- Operational sha256 changes → STOP.

## Risks

- The RK1 acoustic loop body may need slightly different inputs (e.g., the "candidate" state vs "save" state) than RK2/RK3 per WRF source. Cite carefully.
- A "fix" that simply duplicates the RK2 body at RK1 may have wrong state inputs; verify against WRF.

## Handoff Requirements

When all parity gates PASS + B6 regression clean + worker-report committed: `/exit`. After d2h-inside-loop-fix also closes, M6b RETRY V3 dispatches.

Time budget: **30-60 min** (small localized fix).
