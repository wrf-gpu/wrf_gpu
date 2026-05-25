# Sprint Contract — M6b Fix: advance_mu_t Commit + 3 Secondary Defects

## Objective

Two independent diagnostics (opus sanity check `tester/opus/m6b-rk1-sanity-check` + codex real-IC bisection `worker/gpt/m6b-real-ic-bisection`) named the same primary defect:

**`operational_mode.py:_wrf_small_step_acoustic` (lines 307-313)** computes `advance_mu_t_wrf(inputs)` and stores results in `OperationalCarry` scratch, but builds `next_state = state.replace(w=w_solved)` — **leaving `advanced["mu"]`, `advanced["theta"]`, `advanced["mudf"]` (and likely `muts`, `muave`, `ww`, `t_2ave`, `ph_tend`) out of the committed prognostic State**.

Worst observed delta on real Gen2 IC step-1 acoustic substep 1: **theta = 5231.4 K**.

Plus 3 secondary defects flagged by the sanity check:
- **W coefficients not recomputed per substep** (validation does; operational doesn't)
- **dt_sub mismatch**: operational uses dt/3 (RK stages); validation uses dt/10 (acoustic substeps)
- **ph_tend formula mismatch** vs WRF source

Apply all 4 fixes with WRF source citation per defect. Critical: the carry-fix worker's previous report noted "direct theta/mu promotion attempt was tested locally and backed out because it made the 70-second probe nonfinite." The previous failure was likely due to **defect 1 fixed in isolation while defects 2-4 remained broken** (e.g., wrong dt_sub feeding back into mu update making it diverge). This sprint fixes all 4 together to break that feedback.

## Non-Goals

- NO modifications to validation-mode code (locked).
- NO modifications to operational `wrf.exe`.
- NO sanitizer / clamps.
- NO 1h forecast in this sprint (10-step probe + step-1 parity comparison only).
- NO remote push.
- NO imports of validation-only helpers into operational_mode.py.
- NO speculative fixes — only the 4 defects named.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_mu_t_fix` on branch `worker/gpt/m6b-fix-advance-mu-t-commit`.

Write-only:
- `src/gpuwrf/runtime/operational_mode.py` — apply 4 fixes (primary + 3 secondary)
- `src/gpuwrf/runtime/operational_state.py` — extend OperationalCarry if needed for promotion plumbing
- `tests/test_m6b_fix_advance_mu_t_commit.py` (NEW)
- `.agent/sprints/2026-05-25-m6b-fix-advance-mu-t-commit/` — proofs + worker-report

Read-only:
- `src/gpuwrf/dynamics/coupled_step.py` + `acoustic_loop.py` + `mu_t_advance.py` (validation reference — read, don't import)

## Inputs (mandatory)

1. This sprint contract
2. `.agent/sprints/2026-05-25-m6b-rk1-sanity-check/sanity_memo.md` (4-defect localization with file:line)
3. `.agent/sprints/2026-05-25-m6b-real-ic-bisection/worker-report.md` (runtime confirmation of primary defect)
4. `.agent/sprints/2026-05-25-m6b-fix-carry-expansion/worker-report.md` (the carry-fix worker's note that "direct theta/mu promotion attempt was tested locally and backed out because it made the 70-second probe nonfinite" — UNDERSTAND why)
5. `src/gpuwrf/runtime/operational_mode.py:_wrf_small_step_acoustic:307-313` (the smoking gun)
6. WRF source citations (from real-IC bisection):
   - `dyn_em/solve_em.F:1472-1475` (RK1 small step for RK3)
   - `dyn_em/solve_em.F:3435-3452` (advance_mu_t inside small_steps)
   - `dyn_em/module_small_step_em.F:1102-1108` (MU/MUDF/MUTS/MUAVE in place)
   - `dyn_em/module_small_step_em.F:1141-1171` (theta in place)

## Acceptance Criteria

### Stage 1 — Primary fix: promote advance_mu_t outputs (MANDATORY)

In `_wrf_small_step_acoustic`:
- Replace `mu_new = state.mu_perturbation` with `mu_new = advanced["mu"]` (or appropriate WRF-shaped variable)
- Promote `advanced["theta"]`, `advanced["mudf"]`, `advanced["muts"]`, `advanced["muave"]`, `advanced["ww"]`, `advanced["t_2ave"]`, `advanced["ph_tend"]` to the prognostic state via `state.replace(...)`
- Cite WRF source `module_small_step_em.F:1102-1108` + `:1141-1171` per field

### Stage 2 — Secondary fix: per-substep W coefficient recomputation (MANDATORY if applicable)

The validation `acoustic_substep_wrf` recomputes W coefficients on every substep iteration. Operational currently doesn't (per sanity check Part 3).

Add per-substep W-coefficient recomputation. Cite the WRF source where validation does it.

### Stage 3 — Secondary fix: dt_sub correctness (MANDATORY)

dt_sub should be `dt / acoustic_substeps`, not `dt / RK_stages`. Fix the dt computation. Cite WRF.

### Stage 4 — Secondary fix: ph_tend formula (MANDATORY)

Match validation's ph_tend formula per WRF source. Cite the line.

### Stage 5 — Step-1 parity probe (MANDATORY)

`scripts/m6b_real_ic_operational_compare.py --gen2-run-id 20260521_18z_l3_24h_20260522T072630Z --steps 1`. Acceptance: max-abs delta on theta/mu/mudf/ww/muave/muts/ph_tend/t_2ave < 1e-10 (ideally FP64 ULP).

If FAIL: name the next remaining defect (5th); document; route to follow-up.

Capture: `proof_step1_parity_after_fix.json`.

### Stage 6 — 10-step probe (MANDATORY)

`scripts/m6b_real_ic_operational_compare.py --gen2-run-id 20260521_18z_l3_24h_20260522T072630Z --steps 10`. Acceptance: max-abs delta bounded < 1e-8 across all fields, no nonfinite.

Capture: `proof_step10_probe.json`.

### Stage 7 — 70-step probe (MANDATORY — the previous failure point)

Run for 70 timesteps (the original carry-fix worker's "nonfinite at 70s" point). Acceptance: bounded theta within per-level bounds, no nonfinite, max wind plausible.

Capture: `proof_step70_probe.json`.

### Stage 8 — B6 validation regression (MANDATORY)

`python scripts/m6b6_coupled_step_compare.py --tier golden`. Acceptance: still 0.0 bitwise.

### Stage 9 — No regression

```bash
pytest tests/test_m6x_*.py tests/test_m3_transfer_audit.py tests/test_m6b0_*.py tests/test_m6b0r_*.py tests/test_m6b1_*.py tests/test_m6b2_*.py tests/test_m6b3_*.py tests/test_m6b_hygiene_*.py tests/test_m6b4_*.py tests/test_m6b5_*.py tests/test_m6b6_*.py tests/test_m6_operational_*.py tests/test_m6_perf_*.py tests/test_m6b_carry_expansion_*.py tests/test_m6b_d2h_warmed_*.py tests/test_m6b_real_ic_bisection.py tests/test_m6b_fix_advance_mu_t_commit.py -v
```

### Stage 10 — Worker report

`worker-report.md`: per-defect diff summary (4 sections), per-step parity tables, B6 regression status, files changed, **M6b honest 1h RETRY recommendation** (`READY-FOR-M6b-HONEST-1H-V3` if all gates pass).

## Validation Commands

```bash
cd /tmp/wrf_gpu2_mu_t_fix
taskset -c 0-3 python scripts/m6b_real_ic_operational_compare.py --gen2-run-id 20260521_18z_l3_24h_20260522T072630Z --steps 1 2>&1 | tee .agent/sprints/2026-05-25-m6b-fix-advance-mu-t-commit/proof_step1_parity.txt
taskset -c 0-3 python scripts/m6b_real_ic_operational_compare.py --gen2-run-id 20260521_18z_l3_24h_20260522T072630Z --steps 10 2>&1 | tee .agent/sprints/2026-05-25-m6b-fix-advance-mu-t-commit/proof_step10_probe.txt
taskset -c 0-3 python scripts/m6b_carry_expansion_probe.py --runs 1 --duration-s 70 2>&1 | tee .agent/sprints/2026-05-25-m6b-fix-advance-mu-t-commit/proof_step70_probe.txt
taskset -c 0-3 python scripts/m6b6_coupled_step_compare.py --tier golden 2>&1 | tee .agent/sprints/2026-05-25-m6b-fix-advance-mu-t-commit/proof_b6_regression.txt
pytest <full test list> -v 2>&1 | tee .agent/sprints/2026-05-25-m6b-fix-advance-mu-t-commit/proof_no_regression.txt
```

## Kill Gates

- Step-1 parity still > 1e-10 → name the 5th defect; escalate.
- 10-step probe diverges → fix is incomplete; escalate.
- 70-step probe goes nonfinite (the previous failure point) → understand why; if defect 2/3/4 was the missing piece, document; if a NEW 5th defect, escalate.
- B6 regression → REJECT, revert.
- Operational sha256 changes → STOP.

## Risks

- **Previous attempt went nonfinite at 70s with only defect 1 fixed.** This time we fix all 4 in one pass to break the feedback. If it still goes nonfinite, the feedback must come from one of defects 2-4 individually — bisect by fixing one at a time.
- The mu/theta promotion may interact with the validation's bitwise B6 result (since validation goes through `coupled_step.py`, not operational_mode.py — they should be independent code paths). Verify B6 stays 0.0.

## Handoff Requirements

When all 4 gates PASS + worker-report committed: `/exit`. Manager dispatches **M6b honest 1h V3** (the actual M6b acceptance, with the operational mode finally complete).

Time budget: **45-90 min**. 4 defects, mostly mechanical with strong diagnostics already in tree.
