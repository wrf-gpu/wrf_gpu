# Sprint Contract — M6.x ADR-023 Production-Grade Reviewer (Opus 4.7)

## Objective

Per the sprint-lifecycle hard rule, every code/governance sprint requires an independent Opus 4.7 reviewer pass before close. The production-grade ADR-023 implementation landed on main as merge `f4b04af` with all four production-gate tests GREEN, MPAS slice trajectory RMSE driven from 38.7% to 1.69%, epssm sweep complete, mu_continuity un-gated. ADR-023 is currently PROPOSED.

This sprint is the **binding reviewer pass**. The reviewer reads the production-grade branch + ADR-023 + all proof files, runs independent spot-checks, and returns one of:
- `ACCEPT` — promote ADR-023 from PROPOSED to ACCEPTED on main
- `ACCEPT-WITH-REQUIRED-FIXES` — list the fixes; ADR-023 stays PROPOSED until follow-up sprint closes them
- `REJECT` — substantive correctness, anti-tautology, or transfer-audit issue; explain

## Non-Goals

- No code edits. Read-only.
- No re-implementation. The reviewer reads what the worker shipped.
- No re-arguing ADR-021 vs ADR-022 vs ADR-023 architecture — that pivot decision is already ratified by the round-2 critic.
- No sub-sprints. Single-shot.

## File Ownership

Write-only to this sprint folder. Commit your verdict on branch `reviewer/opus/m6x-adr023-production-grade-reviewer`.

## Inputs

Required reading:
- **Production-grade branch tip**: `worker/gpt/m6x-adr023-production-grade @ 0a05159`. Full diff vs `9f19960`.
- `.agent/decisions/ADR-023-conservative-column-solver.md` — current state with worker-inlined F2/F5/F6/F7/F9 closures
- `.agent/sprints/2026-05-23-m6x-adr023-production-grade/worker-report.md` — worker's narrative
- `.agent/sprints/2026-05-23-m6x-adr023-production-grade/proof_*.txt` — all 7 proof files
- `.agent/sprints/2026-05-23-m6x-adr023-production-grade/proof_warm_bubble_production.json`
- `.agent/sprints/2026-05-23-m6x-adr023-three-way-critic/reviewer-report.md` — critic's 10 findings; verify which are closed
- `.agent/sprints/2026-05-23-m6x-adr023-mpas-column-slice-oracle/worker-report.md` — slice baseline (38.7% RMSE)
- `src/gpuwrf/dynamics/acoustic_wrf.py` — production operator
- `src/gpuwrf/dynamics/vertical_implicit_solver.py` — solver primitive
- `tests/test_m6x_adr023_production_grade.py` — production-gate tests
- `src/gpuwrf/validation/mpas_oracles/mpas_column_slice.py` — what the operator is being compared to (do NOT modify)
- WRF source `module_small_step_em.F` lines 619-1597 (canonical)
- MPAS source `mpas_atm_time_integration.F` lines 1589-2208 (slice reference)

## Acceptance Criteria

`reviewer-report.md` containing seven labelled sections:

1. **§1 Re-run spot checks.** Re-execute the four key proof commands independently. Capture outputs in this sprint folder as `spot_*.txt`:
   - `pytest tests/test_m6x_adr023_production_grade.py -v`
   - `pytest tests/test_m6x_mpas_column_slice_oracle.py tests/test_m6x_vertical_acoustic_oracle.py tests/test_m6x_adr023_column_solver.py tests/test_m6x_c2_acoustic.py -v`
   - `python scripts/m6_warm_bubble_test.py --output /tmp/reviewer_wb.json`
   - `pytest tests/test_m3_transfer_audit.py tests/test_m6x_c2_acoustic.py::test_acoustic_scan_jaxpr_has_scan_and_no_host_callbacks -v`
   - Confirm: 4/4 production gate PASS; 19/19 no-regression PASS; warm-bubble `PASS_WARM_BUBBLE_600S` with `w_max ∈ [5, 10]`; 5/5 transfer audit PASS.

2. **§2 Critic findings closure audit.** For each of the 10 critic findings F1-F10:
   - State whether the worker's evidence closes it.
   - Cite the closing artifact (file:line or proof file).
   - If a finding is still open or only partially closed, label it.

3. **§3 Anti-tautology audit.** Confirm:
   - The MPAS slice oracle is genuinely independent (NumPy literal port, not a JAX integrator wearing an MPAS label).
   - The production operator is NOT importing or calling the slice oracle in its production path.
   - The R7 analytic oracle and the slice oracle are file-disjoint from `acoustic_wrf.py`.
   - The 1.69% trajectory RMSE is computed against the slice, not against itself.
   - Spec-gaming patterns from M5 (verifiability triple) are NOT present.

4. **§4 Forbidden-move audit.** Confirm worker did NOT:
   - Expand `AcousticScanCarry` beyond the 6-leaf form
   - Add a Newton outer loop
   - Modify any oracle test or oracle module
   - Modify c2-A2 horizontal PGF or `mu_continuity_tendency`
   - Self-promote ADR-023 from PROPOSED to ACCEPTED (per contract, that's your call)
   - Introduce host/device transfers

5. **§5 Open risks.** Specifically address:
   - Launch count growth 20 → 67 (3.3×). Is this acceptable for an interim production-grade ADR, with optimization deferred to a post-M6 sprint? Cite the project's launch-count rule (if any).
   - 48 device-to-device memcpy calls. Not host transfers, but a real GPU memory-pressure indicator.
   - `epssm=0.3` failed R7 by 4.57% but had the best slice RMSE — does the chosen default 0.1 trade-off correctly?

6. **§6 Verdict** (exactly one):
   - `ACCEPT` — promote ADR-023 PROPOSED → ACCEPTED on next manager merge. List the conditions you treat as already satisfied.
   - `ACCEPT-WITH-REQUIRED-FIXES` — list the fixes (file:line, severity). ADR-023 stays PROPOSED.
   - `REJECT` — substantive correctness, anti-tautology, or transfer-audit issue; explain in detail.

7. **§7 Open questions for the manager.** Anything the manager should resolve before dispatching the next rung (1h d02 boundary replay or 24h/72h Gen2 RMSE).

## Required commit step

When `reviewer-report.md` is written:
```bash
cd /tmp/wrf_gpu2_review_prod
git switch -c reviewer/opus/m6x-adr023-production-grade-reviewer
git add .agent/sprints/2026-05-23-m6x-adr023-production-grade-reviewer/reviewer-report.md spot_*.txt
git commit -m "[ADR-023 production-grade reviewer] <verdict>"
```

The branch is pre-created by the manager so you should already be on it.

## Validation Commands

The §1 spot-check commands above. Output goes to this sprint folder as `spot_*.txt`.

## Performance Metrics

N/A — reviewer sprint.

## Proof Object

- `reviewer-report.md` (3000-6000 words)
- `spot_*.txt` for each re-run command
- Verdict committed on `reviewer/opus/m6x-adr023-production-grade-reviewer`

Time budget: **45-90 min**.

## Risks

- **Worker-reviewer same-AI risk**: the production-grade worker was codex; the reviewer must be Opus 4.7 (different blind spots). This sprint is dispatched on Opus.
- **Surface confirmation bias**: 4/4 production gate PASS is suspicious if the gates were authored by the worker. The reviewer must verify the gates are non-tautological — see §3.
- **Skipping §3 anti-tautology audit** is the highest-risk shortcut. Spend time there.

## Handoff Requirements

When the commit lands and `reviewer-report.md` is on disk, type `/exit` as a slash command. Wrapper watchdog fires `AGENT REPORT [reviewer / m6x-adr023-production-grade-reviewer / opus] exit=<ec>` to the manager pane.
