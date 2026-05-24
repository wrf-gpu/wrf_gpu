# Sprint Contract — M6B0-R Reproducer Audit (opus tester)

## Objective

Independently re-run M6B0-R's Stage 5 (the JAX-vs-WRF-Python-reproduction comparator) on the M6B0-R-emitted savepoints **without any modification to the JAX side or the comparator**. Verify the `PARITY-DEFECT-LOCALIZED` verdict is reproducible. Also sanity-check the M6B0-R Python-reproduction of `calc_coef_w` against the canonical WRF Fortran source (`module_small_step_em.F:570-652`).

This is parallel insurance: the DEFECT-ANALYSIS lane will assume the M6B0-R verdict is correct and propose a fix. If the verdict were a measurement artifact, the fix would chase a phantom. This sprint catches that.

## Non-Goals

- NO modifications to JAX code, comparator code, or savepoint code.
- NO modifications to operational `wrf.exe`.
- NO new sprint dispatch.
- NO writing to `external/wrf_savepoint_patch/` (RELINK lane), `src/gpuwrf/dynamics/` (DEFECT-ANALYSIS lane), or `src/gpuwrf/validation/` (locked).
- Do NOT propose a fix; that is DEFECT-ANALYSIS's job. Your job is to **verify**.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_reprod` on branch `tester/opus/m6b0r-reproducer-audit`.

Write-only:
- `.agent/sprints/2026-05-24-m6b0r-reproducer-audit/audit_memo.md` (deliverable)
- `.agent/sprints/2026-05-24-m6b0r-reproducer-audit/proof_*.txt`, `proof_*.json`

Read-only everywhere else.

## Inputs

1. `.agent/sprints/2026-05-24-m6b0r-reproducer-audit/sprint-contract.md` (this)
2. `.agent/sprints/2026-05-24-m6b0r-real-fortran-emission/worker-report.md`
3. `.agent/sprints/2026-05-24-m6b0r-real-fortran-emission/proof_real_coefficient_parity.json`
4. M6B0-R savepoints (in `.agent/sprints/2026-05-24-m6b0r-real-fortran-emission/savepoints/` if committed; otherwise re-extract via `scripts/m6b0r_wrf_savepoint_extract.py`)
5. `scripts/m6b0r_jax_vs_wrf_compare.py` (the comparator — read it; understand its Python `calc_coef_w` reproduction)
6. WRF source: `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/dyn_em/module_small_step_em.F:570-652`
7. `src/gpuwrf/validation/tolerance_ladder.json`

## Acceptance Criteria

### Part 1 — Reproduce M6B0-R Stage 5 (MANDATORY)

Run the comparator unchanged on the existing savepoints. Verify the worst-case deltas reported in `proof_real_coefficient_parity.json` (a/alpha/gamma) reproduce within numerical noise. If they don't, document the discrepancy and STOP — that's a critical finding.

Capture proof: `proof_reproduce_stage5.txt` + `proof_reproduce_stage5.json`.

### Part 2 — Sanity-check the Python `calc_coef_w` reproduction (MANDATORY)

Read `scripts/m6b0r_jax_vs_wrf_compare.py`'s Python implementation of `calc_coef_w`. Compare line-by-line against WRF source `:570-652`. Produce a discrepancy table:
- Each WRF line / variable
- The Python reproduction's corresponding line
- Match / Mismatch / Missing

If the reproduction has bugs, document them with WRF source citation. **Do NOT fix** — flag to manager.

Capture: `proof_python_reproduction_audit.md`.

### Part 3 — Independent recomputation on Tier-1 column (MANDATORY)

Write a tiny standalone script in your worktree (NOT committed to scripts/ — keep in `.agent/sprints/2026-05-24-m6b0r-reproducer-audit/`) that:
- Loads one column's `calc_coef_w_pre` savepoint
- Computes `a`, `alpha`, `gamma` per the WRF source equation (your own reading, not the M6B0-R reproduction)
- Compares against both:
  - M6B0-R Python-reproduction output
  - JAX implementation output
- Reports which (if any) of the three agrees with your independent computation

Capture: `proof_independent_recomputation.json`.

### Part 4 — Verdict memo

`audit_memo.md` answers:
1. Is M6B0-R's `PARITY-DEFECT-LOCALIZED` verdict reproducible? (YES / NO with evidence)
2. Is the M6B0-R Python `calc_coef_w` reproduction faithful to WRF source? (YES / PARTIAL / NO with citations)
3. From your independent recomputation, who is "right" for a/alpha/gamma — JAX, M6B0-R reproduction, your independent reading, or neither?
4. GO / NO-GO for the DEFECT-ANALYSIS lane to proceed with its planned fix (GO = M6B0-R verdict is trustworthy enough to act on; NO-GO = re-baseline before fixing).

### Part 5 — No regression

`pytest --collect-only 2>&1 | tail -3` — verify no test changes.

## Validation Commands

```bash
cd /tmp/wrf_gpu2_reprod
python scripts/m6b0r_jax_vs_wrf_compare.py --operator calc_coef_w --tier all 2>&1 | tee .agent/sprints/2026-05-24-m6b0r-reproducer-audit/proof_reproduce_stage5.txt
# Plus your independent recomputation script
pytest --collect-only 2>&1 | tail -3 | tee .agent/sprints/2026-05-24-m6b0r-reproducer-audit/proof_no_touch.txt
```

## Performance Metrics

N/A — opus probe.

## Proof Object

- `audit_memo.md` (GO/NO-GO)
- `proof_*.txt`, `proof_*.json` per stages
- Branch `tester/opus/m6b0r-reproducer-audit`

Time budget: **60–120 min**.

## Risks

- If the M6B0-R Python reproduction is WRONG and DEFECT-ANALYSIS lane assumes it is correct, the JAX side gets "fixed" toward a wrong target. This sprint is the firewall.
- Independent recomputation might agree with JAX (defect is in M6B0-R reproduction), with M6B0-R reproduction (defect is in JAX, as M6B0-R says), or with neither (multiple bugs). All three are valid outcomes; report what you find.

## Handoff Requirements

When `audit_memo.md` + proofs committed on branch `tester/opus/m6b0r-reproducer-audit`: stop. Manager reads memo before merging DEFECT-ANALYSIS lane's fix.
