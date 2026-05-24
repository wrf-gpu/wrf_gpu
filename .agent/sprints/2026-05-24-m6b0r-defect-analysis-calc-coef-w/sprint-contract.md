# Sprint Contract — M6B0-R Defect Analysis: `calc_coef_w` a/α/γ Mismatch

## Objective

M6B0-R (commit `worker/gpt/m6b0r-real-fortran-emission`) localized `PARITY-DEFECT-LOCALIZED` for `a`, `alpha`, `gamma` in `calc_coef_w` on all 3 tiers, sanitizer-off. The observed worst-case deltas (`a=259.66`, `alpha=0.995`, `gamma=0.479` against `1e-11` thresholds) are not numerical drift — they are formulation/sign/unit/staggering mismatches.

This sprint is a **forensic deep-dive**: read WRF Fortran `calc_coef_w` at `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/dyn_em/module_small_step_em.F:570-652` line by line, read the JAX equivalent in the current dycore, identify EXACTLY what differs, write a minimal localized fix, and demonstrate the fix improves parity on the existing M6B0-R proof JSONs.

## Non-Goals

- NO multi-operator changes. `calc_coef_w` only.
- NO new clamps/dampings/sanitizer adjustments.
- NO modifications to operational `wrf.exe`. Pre/post sha256 enforced.
- NO modifications to comparator infrastructure (`src/gpuwrf/validation/*`). Already validated.
- NO touching `external/wrf_savepoint_patch/` (that's RELINK lane's territory).
- NO 1h forecast or warm-bubble runs.
- NO regression of any other operator's behavior.
- NO remote push.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_defect` on branch `worker/gpt/m6b0r-defect-analysis-calc-coef-w`.

Write-only:
- `src/gpuwrf/dynamics/acoustic_wrf.py` — **localized fix only**, with WRF source citations for every change (file:line ranges) in commit messages and inline comments where the fix is non-obvious.
- `tests/test_m6b0r_calc_coef_w_fix.py` (NEW) — regression test pinning the new behavior to the M6B0-R savepoint deltas (must improve, must not regress other tiers).
- `.agent/sprints/2026-05-24-m6b0r-defect-analysis-calc-coef-w/defect_analysis.md` — the forensic report
- `.agent/sprints/2026-05-24-m6b0r-defect-analysis-calc-coef-w/proof_*.txt`, `proof_*.json` — before/after deltas
- `.agent/sprints/2026-05-24-m6b0r-defect-analysis-calc-coef-w/worker-report.md`

Read-only:
- `external/wrf_savepoint_patch/` (RELINK lane is editing this)
- `src/gpuwrf/validation/` (locked from M6B0-R)
- `.agent/sprints/2026-05-24-m6b0r-real-fortran-emission/` (inputs only)

## Inputs (mandatory)

1. `.agent/sprints/2026-05-24-m6b0r-defect-analysis-calc-coef-w/sprint-contract.md` (this)
2. `.agent/sprints/2026-05-24-m6b0r-real-fortran-emission/worker-report.md`
3. `.agent/sprints/2026-05-24-m6b0r-real-fortran-emission/proof_real_coefficient_parity.json` (the defect numbers)
4. `.agent/sprints/2026-05-24-m6b0r-real-fortran-emission/savepoints/golden/*` (the golden-slice savepoints — replay them)
5. **WRF source**: `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/dyn_em/module_small_step_em.F:570-652` (the canonical `calc_coef_w` SUBROUTINE)
6. JAX equivalent: `src/gpuwrf/dynamics/acoustic_wrf.py` (search for the function that computes the vertical-implicit-solve coefficients; verify with `grep -n 'cofwz\|cofwr\|cofwt\|coftz\|coef_w' src/gpuwrf/dynamics/*.py`)
7. `.agent/decisions/source_mining_operator_table.md` (operator term provenance)
8. `src/gpuwrf/validation/tolerance_ladder.json` (the ladder Stage 5 used)
9. `scripts/m6b0r_jax_vs_wrf_compare.py` (the comparator to re-run for evidence)

## Acceptance Criteria

### Stage 1 — Read WRF Fortran `calc_coef_w` and document every term (MANDATORY)

In `defect_analysis.md`, produce a per-line table for `module_small_step_em.F:570-652`:
- Variable name in WRF
- WRF's computation (cite line)
- Units
- Stagger
- Dependencies (which prior outputs feed in)
- JAX equivalent location (`acoustic_wrf.py:LINE`)
- Match / Mismatch / Missing

### Stage 2 — Identify the formulation gap (MANDATORY)

In `defect_analysis.md`, name the ONE OR MORE specific discrepancies:
- Equation form
- Sign convention
- Unit interpretation (`g`, `Rd`, `cv`, etc. — WRF source uses module-level constants from `module_model_constants.F`; verify the JAX side uses the same numerical values)
- Staggering interpretation (mass-point vs eta-half vs eta-full)
- Coefficient assembly order (e.g., WRF builds `cofwz` first then `cofwr` uses it; JAX may have built them out of order)

Cite WRF source line for every claim. Show the offending JAX line for every claim.

### Stage 3 — Minimal localized fix (MANDATORY)

Edit `src/gpuwrf/dynamics/acoustic_wrf.py` to apply the smallest possible change that aligns with WRF source. Multiple small changes allowed if each is cited and tested independently. NO new constants from thin air. NO new stabilizers. NO new clamps.

### Stage 4 — Re-run the comparator (MANDATORY)

Re-run `scripts/m6b0r_jax_vs_wrf_compare.py --operator calc_coef_w --tier all` against the existing M6B0-R savepoints (do NOT re-extract; re-use the bundled savepoints from the merged manager branch).

Acceptance bar (Tier-1 column, sanitizer-off):
- `a`, `alpha`, `gamma` all within 1e-6 absolute tolerance vs WRF-shape Python reproduction.
- If 1e-6 cannot be reached, document the residual and route to a follow-on M6B0-R-FIX-V2 sprint.

If parity DOES achieve full match (within ladder), the sprint outcome is `FIRST-OPERATOR-PARITY-ACHIEVED`. This unblocks M6B1 (`advance_mu_t` parity).

Capture proofs: `proof_calc_coef_w_before.json`, `proof_calc_coef_w_after.json`, `proof_calc_coef_w_delta.txt`.

### Stage 5 — Regression test (MANDATORY)

Write `tests/test_m6b0r_calc_coef_w_fix.py` that:
- Loads the M6B0-R golden-slice savepoint pre-state
- Runs the new JAX `calc_coef_w`
- Asserts `a`, `alpha`, `gamma` match the WRF-shape Python reproduction within the ladder tolerance
- Asserts no other operator's existing test regresses

### Stage 6 — No regression (MANDATORY)

```bash
pytest tests/test_m6x_*.py tests/test_m3_transfer_audit.py tests/test_m6b0_*.py tests/test_m6b0r_*.py tests/test_m6b0r_calc_coef_w_fix.py -v
```
All PASS. The previously failing parity test (if any) now PASSES with the fix applied.

### Stage 7 — Worker report

`worker-report.md`: defect mechanism (1-2 paragraphs), the fix (file:line diff summary + WRF source citation), before/after deltas, test results, files changed, risks, handoff to M6B1 (or to M6B0-R-FIX-V2 if 1e-6 not reached).

## Validation Commands

```bash
cd /tmp/wrf_gpu2_defect
python scripts/m6b0r_jax_vs_wrf_compare.py --operator calc_coef_w --tier all 2>&1 | tee .agent/sprints/2026-05-24-m6b0r-defect-analysis-calc-coef-w/proof_calc_coef_w_after.txt
pytest tests/test_m6x_*.py tests/test_m3_transfer_audit.py tests/test_m6b0_*.py tests/test_m6b0r_*.py tests/test_m6b0r_calc_coef_w_fix.py -v 2>&1 | tee .agent/sprints/2026-05-24-m6b0r-defect-analysis-calc-coef-w/proof_no_regression.txt
```

## Performance Metrics

N/A — correctness sprint.

## Kill Gates

- Cannot identify the formulation gap in ≤2 hours of reading → escalate: dispatch a parallel codex critic or request external WRF expert review.
- Fix is non-localized (touches ≥3 unrelated files) → reject; the defect must localize to `calc_coef_w` or its direct dependencies.
- Test regresses another operator → revert; document in worker-report and route as a wider sprint.

## Risks

- The JAX implementation may have multiple coefficient bugs simultaneously — fixing only one might not bring a/α/γ within tolerance. Allow worker to fix multiple in this sprint **if** each is independently cited.
- The Python-WRF reproduction the M6B0-R worker built (in `scripts/m6b0r_jax_vs_wrf_compare.py`) might itself have a bug. Worker MUST sanity-check the reproduction against the WRF Fortran source before assuming the JAX side is wrong. If the reproduction is wrong, fix the reproduction instead and document in worker-report.

## Handoff Requirements

When all proofs + worker-report.md committed on branch `worker/gpt/m6b0r-defect-analysis-calc-coef-w`: `/exit`. Manager merges + (if `FIRST-OPERATOR-PARITY-ACHIEVED`) dispatches M6B1 + (if `RESIDUAL-DEFECT`) dispatches M6B0-R-FIX-V2.

## Failure modes the manager will reject

- "Fix" without WRF source citation.
- Adding new constants/factors without WRF source citation.
- Hidden adjustments to the comparator or tolerance ladder.
- Skipping Stage 4 (re-running the comparator).
- Multi-suspect mass changes.
- Modifying operational `wrf.exe`.
