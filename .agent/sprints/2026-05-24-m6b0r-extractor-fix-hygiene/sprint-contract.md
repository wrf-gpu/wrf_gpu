# Sprint Contract — M6B0-R Extractor Top-Row Hygiene (opus tester)

## Objective

Reproducer-audit flagged two top-row bugs in `scripts/m6b0r_wrf_savepoint_extract.py` `_wrf_calc_coef_w` (lines 122-163):

1. **Line 138**: `lid_flag = 1.0` hardcoded; WRF spec is `lid_flag = 0 if top_lid else 1`. For the current Canary d02 runs the namelist has `TOP_LID=F` so `lid_flag=1` happens to be correct — but this is latent: re-extracting under a different namelist would silently produce wrong top-row coefficients.

2. **Line 142**: `denom_top` uses `c1f[nz]` but Fortran wants `c1f(kde-1)` for the top `a` row. The same `denom_top` does happen to be correct for `b_top` (which wants `c1f(kde)`), so currently the bug cancels in one case but is wrong in the other.

This sprint converts the two latent bugs to canonical handling, regenerates the affected savepoint slices, re-runs the comparator, and demonstrates that `calc_coef_w` parity (achieved at `ac252e8`) still holds with the corrected extractor.

## Non-Goals

- NO changes to operational `wrf.exe`.
- NO changes to `src/gpuwrf/dynamics/`, `src/gpuwrf/validation/`, or any JAX side.
- NO changes to the comparator beyond what is mechanically required if the extractor's output shape changes.
- NO multi-operator changes.
- NO 1h forecast.
- NO sub-sprint dispatch.
- NO remote push.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_extfix` on branch `tester/opus/m6b0r-extractor-fix-hygiene`.

Write-only:
- `scripts/m6b0r_wrf_savepoint_extract.py` — fix lines 138 + 142, cite WRF source
- `tests/test_m6b0r_extractor_top_row.py` (NEW) — regression that fails if either bug returns
- `.agent/sprints/2026-05-24-m6b0r-extractor-fix-hygiene/` — proofs + memo

Read-only everywhere else.

## Inputs

1. `.agent/sprints/2026-05-24-m6b0r-extractor-fix-hygiene/sprint-contract.md` (this)
2. `.agent/sprints/2026-05-24-m6b0r-reproducer-audit/audit_memo.md` + `proof_python_reproduction_audit.md` (the bug report)
3. `.agent/sprints/2026-05-24-m6b0r-defect-analysis-calc-coef-w/worker-report.md` (the prior fix)
4. WRF source: `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/dyn_em/module_small_step_em.F:570-652`
5. `scripts/m6b0r_wrf_savepoint_extract.py` (file under audit)

## Acceptance Criteria

### Stage 1 — Bug 1: `lid_flag` conditional (MANDATORY)

Change `lid_flag = 1.0` to read the namelist (or metadata) `top_lid` attribute and set `lid_flag = 0.0 if top_lid else 1.0`. Cite WRF source line for the conditional.

### Stage 2 — Bug 2: `denom_top` index (MANDATORY)

Audit WRF source: the top `a` row uses `c1f(kde-1)` (one face below model top), the top `b` row uses `c1f(kde)` (the top face). Either compute both indices explicitly or document why a single `denom_top` was acceptable for the prior fix and add a test asserting the relationship.

### Stage 3 — Regression test (MANDATORY)

`tests/test_m6b0r_extractor_top_row.py`:
- Test that with `top_lid=True` metadata, `lid_flag` is 0
- Test that with `top_lid=False` metadata, `lid_flag` is 1
- Test that top-row `a` uses `c1f(kde-1)` not `c1f(kde)`
- Test that top-row `b` uses `c1f(kde)` as before

### Stage 4 — Re-run calc_coef_w parity comparator (MANDATORY)

Re-run `scripts/m6b0r_jax_vs_wrf_compare.py --operator calc_coef_w --tier all`. Acceptance:
- For the current Canary d02 namelist (`TOP_LID=F`), parity must still hold (a/alpha/gamma worst delta = 0) because `lid_flag=1` is unchanged for this namelist.
- The `denom_top` fix may slightly shift the top-row `a` value; if it does, the new value is canonical and the parity test must still pass (since both extractor and JAX use the corrected formula).

If parity DOES regress (i.e., the JAX side was matching the buggy extractor at the top row): document the regression, do NOT fix JAX in this sprint, route to a follow-up M6B0-R-JAX-TOP-ROW-FIX sprint with a 3-line localized JAX fix.

Capture: `proof_calc_coef_w_after_extractor_fix.txt` + `.json`.

### Stage 5 — No regression (MANDATORY)

```bash
pytest tests/test_m6x_*.py tests/test_m3_transfer_audit.py tests/test_m6b0_*.py tests/test_m6b0r_*.py tests/test_m6b0r_calc_coef_w_fix.py tests/test_m6b0r_extractor_top_row.py -v
```

### Stage 6 — Memo

`extractor_fix_memo.md`: bugs fixed, citations, parity result after fix, files changed, GO/NO-GO for M6B1.

## Validation Commands

```bash
cd /tmp/wrf_gpu2_extfix
pytest tests/test_m6b0r_extractor_top_row.py -v 2>&1 | tee .agent/sprints/2026-05-24-m6b0r-extractor-fix-hygiene/proof_new_test.txt
python scripts/m6b0r_jax_vs_wrf_compare.py --operator calc_coef_w --tier all 2>&1 | tee .agent/sprints/2026-05-24-m6b0r-extractor-fix-hygiene/proof_calc_coef_w_after_extractor_fix.txt
pytest tests/test_m6x_*.py tests/test_m3_transfer_audit.py tests/test_m6b0_*.py tests/test_m6b0r_*.py tests/test_m6b0r_calc_coef_w_fix.py tests/test_m6b0r_extractor_top_row.py -v 2>&1 | tee .agent/sprints/2026-05-24-m6b0r-extractor-fix-hygiene/proof_no_regression.txt
```

## Performance Metrics

N/A.

## Risks

- The "fix" may itself reveal that the JAX side was silently matching the buggy extractor — surface this as a regression, do NOT mask it.

## Handoff Requirements

When all proofs + `extractor_fix_memo.md` committed on branch `tester/opus/m6b0r-extractor-fix-hygiene`: stop. Manager merges and decides whether M6B1 proceeds (GO) or needs M6B0-R-JAX-TOP-ROW-FIX first (NO-GO).
