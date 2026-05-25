# Sprint Contract — M6b Fix: Carry Expansion with Bisection Gate

## Objective

Per failure-critic verdict `SPLIT-INTO-TWO-SPRINTS` (commit `tester/opus/m6b-d2h-grep` reviewer-report + critic/codex/m6b-failure-step-back §4): the M6b BLOCKER is operational-mode strict-subset carry **missing WRF scratch families** that validation-mode (M6B6 0.0 bitwise) has. Promote `t_2ave`, `ww`, `muave`, `muts`, `ph_tend`, and `_save` families from Amendment #1 **Undecided** → **Operational-required-with-Tier-4-evidence** (the M6b failure IS the Tier-4 ablation evidence).

Re-run M6b's 10s probe on the 3 pinned Gen2 run-IDs. Acceptance: bounded theta + wind, no validation-mode regression (B6 0.0 must stay 0.0), promotion table written per Amendment #1.

D2H lift is **deferred to a separate small sprint** (now simplified per D2H-grep finding: D2H=53 was Nsight profiling artifact, not real violation; warmed re-capture should show D2H=0).

## Non-Goals

- NO modifications to validation-mode code (acoustic_wrf.py / mu_t_advance.py / tridiag_solve.py / small_step_scratch.py / acoustic_loop.py / dycore_step.py / coupled_step.py — all LOCKED).
- NO modifications to operational `wrf.exe`. Pre/post sha256 (`1ec3815...`).
- NO new physics or new operator semantics.
- NO sanitizer in operational path.
- NO full 1h forecast in this sprint (10s probe only; full 1h is the M6b-retry sprint after this passes).
- NO D2H lift work (separate sprint).
- NO reduction of acoustic substep count from the B-direct baseline for the bisection run (use the SAME substep cadence as M6B6 for the bisection; substep tuning is later).
- NO ADR-026 amendment in this sprint — only the carry-fields promotion table.
- NO remote push.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_carryfix` on branch `worker/gpt/m6b-fix-carry-expansion`.

Write-only:
- `src/gpuwrf/runtime/operational_mode.py` — add carry fields per the promotion table; do NOT import validation-only helpers (acoustic_loop.py / dycore_step.py / coupled_step.py); compose operational variants from the same primitive operators
- `src/gpuwrf/runtime/operational_state.py` (NEW or extend if exists) — operational carry pytree (strict subset + promoted scratch fields with rationale per field)
- `scripts/m6b_carry_expansion_probe.py` (NEW) — 10s probe on 3 Gen2 IDs + B6 golden-aligned single-savepoint comparison
- `tests/test_m6b_carry_expansion_bounded.py` (NEW) — finiteness + bounds assertion on each Gen2 run
- `.agent/sprints/2026-05-25-m6b-fix-carry-expansion/` — proofs + worker-report

Read-only everywhere else.

## Inputs (mandatory)

1. This sprint contract
2. `.agent/sprints/2026-05-25-m6b-failure-critic/reviewer-report.md` (the verdict — read §3 bandwidth + §4 sprint scope)
3. `.agent/sprints/2026-05-25-m6b-d2h-grep/d2h_localization.md` (D2H is artifact; ignore for this sprint)
4. `.agent/sprints/2026-05-25-m6b-honest-1h-canary/worker-report.md` + proof JSONs (the BLOCKER evidence)
5. `.agent/sprints/2026-05-25-m6b3-scratch-state-parity/worker-report.md` (the 6 scratch families + their WRF source citations)
6. `.agent/sprints/2026-05-25-m6b6-coupled-step-parity/worker-report.md` (the 0.0 bitwise baseline that must not regress)
7. `src/gpuwrf/runtime/operational_mode.py` (current operational entry point)
8. `src/gpuwrf/dynamics/small_step_scratch.py` (validation-only — DO NOT import; copy/adapt the scratch update logic)
9. `PROJECT_PLAN.md §14.5.1` Amendment #1 (operational-promotion rule: Undecided → Operational-approved requires Tier-4 evidence)

## Acceptance Criteria

### Stage 1 — Operational state carry expansion (MANDATORY)

`src/gpuwrf/runtime/operational_state.py`: extend operational carry with these promoted fields:

| Field | Promoted from | Rationale |
|---|---|---|
| `t_2ave` | Undecided | M6b failure: theta drift without it (cite proof) |
| `ww` | Undecided | M6b failure: vertical velocity drift without it |
| `muave` | Undecided | M6b failure: mass running average required for acoustic stability |
| `muts` | Undecided | M6b failure: mass at substep required for tendency cadence |
| `ph_tend` | Undecided | M6b failure: geopotential tendency accumulation required |
| `_save` family | Undecided | M6b failure: RK stage transition state required |

Each row cited to the M6b failure proof (per-field reasoning may come from the WRF source where the scratch is consumed).

### Stage 2 — Carry-expansion operational mode (MANDATORY)

Update `operational_mode.run_forecast_operational` to thread the promoted carry through the timestep loop. Do NOT import validation-only helpers; copy/adapt the scratch update math from `small_step_scratch.py` (cite WRF source for each update).

### Stage 3 — Bisection: 10s probe on 3 Gen2 IDs (MANDATORY)

`scripts/m6b_carry_expansion_probe.py`:
- Same 3 pinned Gen2 run-IDs as M6b
- Run operational mode for **10 simulated seconds** (1 timestep at dt=10s, or 6 timesteps at dt=1.667s — pick the cadence matching M6B6's bisection setting)
- Match B-direct acoustic substep count (no reduction yet — bisection is for variable isolation)
- Sanitizer OFF
- Per-run: finiteness, theta bounds (200K-400K), wind bounds (|u|,|v| ≤ 100, |w| ≤ 50)

**Acceptance**: all 3 runs pass bounds at 10s. If FAIL: classify root cause as Hypothesis B (composition bug), STOP, name the first diverging field + cell, escalate.

### Stage 4 — B6 golden-aligned regression (MANDATORY)

Run the M6B6 golden-slice savepoint comparison after applying carry expansion to operational mode. Acceptance: validation-mode `worst delta = 0.0` MUST be unchanged. If operational carry expansion regresses validation-mode bitwise parity by even 1 ULP → REJECT.

### Stage 5 — Promotion table written (MANDATORY, Critic Amendment #1)

`worker-report.md` includes the per-field promotion table (Stage 1) with Tier-4 evidence citation per row. **Default is Undecided**; promotion requires this sprint's evidence.

### Stage 6 — No regression

```bash
pytest tests/test_m6x_*.py tests/test_m3_transfer_audit.py tests/test_m6b0_*.py tests/test_m6b0r_*.py tests/test_m6b1_*.py tests/test_m6b2_*.py tests/test_m6b3_*.py tests/test_m6b_hygiene_*.py tests/test_m6b4_*.py tests/test_m6b5_*.py tests/test_m6b6_*.py tests/test_m6_operational_*.py tests/test_m6_perf_*.py tests/test_m6b_carry_expansion_*.py -v
```

### Stage 7 — Worker report

`worker-report.md`: stages 1-6, promotion table, 10s probe results per Gen2 run, B6 regression status, files changed, **dispatch recommendation for M6b honest 1h re-try** (`READY-FOR-M6b-RETRY` if all gates pass).

## Validation Commands

```bash
cd /tmp/wrf_gpu2_carryfix
taskset -c 0-3 python scripts/m6b_carry_expansion_probe.py --runs 3 --duration-s 10 2>&1 | tee .agent/sprints/2026-05-25-m6b-fix-carry-expansion/proof_10s_probe.txt
taskset -c 0-3 python scripts/m6b6_coupled_step_compare.py --tier golden 2>&1 | tee .agent/sprints/2026-05-25-m6b-fix-carry-expansion/proof_b6_regression.txt
pytest tests/test_m6x_*.py tests/test_m3_transfer_audit.py tests/test_m6b0_*.py tests/test_m6b0r_*.py tests/test_m6b1_*.py tests/test_m6b2_*.py tests/test_m6b3_*.py tests/test_m6b_hygiene_*.py tests/test_m6b4_*.py tests/test_m6b5_*.py tests/test_m6b6_*.py tests/test_m6_operational_*.py tests/test_m6_perf_*.py tests/test_m6b_carry_expansion_*.py -v 2>&1 | tee .agent/sprints/2026-05-25-m6b-fix-carry-expansion/proof_no_regression.txt
```

## Performance Metrics

N/A — correctness recovery sprint.

## Kill Gates

- 10s probe still fails bounds → classify Hypothesis B; STOP; escalate.
- B6 validation regresses (>0 delta) → REJECT.
- Imports validation-only helpers (acoustic_loop / dycore_step / coupled_step) into operational_mode → REJECT per audit risk #6.
- Operational sha256 changes → STOP.

## Risks

- The 6 scratch families have non-trivial initialization (e.g., `_save` is RK-stage-snapshot state). Initialize from real Gen2 wrfout fields at run start.
- Some scratch may need different update frequency (e.g., `t_2ave` is running average across all substeps in one RK stage). Cite WRF source.
- Carry size +40% — verify still fits RTX 5090 memory headroom (~21 GB available); should be fine for d02 (~50 MB carry).

## Handoff Requirements

When 10s probe passes + B6 regression passes + worker-report committed: `/exit`. Manager dispatches:
1. M6b-d2h-warmed-recapture (small opus sprint — re-run Nsight with warm-up, confirm D2H=0)
2. M6b-honest-1h-RETRY (full 1h Canary forecast on 3 Gen2 IDs)

If 10s probe FAILS: escalate; dispatch Hypothesis B bisection sprint.

## Failure modes the manager will reject

- Promotion without Tier-4 citation per row.
- Importing validation-only helpers.
- B6 regression.
- Adding stabilizers/clamps to mask the failure.
- Skipping operational-compat classification table.
