# Sprint Contract — M6B0-R JAX Top-Row Fix Follow-up (opus tester)

**Status:** Pre-drafted 2026-05-24 night. **Dispatch when:** M6B1 closes AND any sprint is about to extract savepoints under a non-Canary namelist (different vertical-coordinate transition).

## Objective

EXTRACTOR-FIX worker flagged latent Bug 2 in JAX: `src/gpuwrf/dynamics/acoustic_wrf.py:635-637,660` carries the same single `top_denom = mass_h[nz-1] * mass_f[nz]` pattern that the extractor had. For Canary hybrid-eta where `c1f[nz-1]=c1f[nz]=0.0`, the bug is silent. Under any namelist where `c1f[nz-1] != c1f[nz]`, the bug becomes observable on top-row `a`.

This sprint is a 3-line localized fix: split `top_denom` into `top_denom_a` (uses `c1f[nz-1]`) and `top_denom_b` (uses `c1f[nz]`). Add a regression test that exercises a synthetic non-zero `c1f[nz-1]` value and asserts the JAX side computes the WRF-canonical top-row coefficients.

## Non-Goals

- NO multi-operator changes. JAX `calc_coef_w_wrf_coefficients` only.
- NO modifications to extractor, comparator, schema.
- NO changes that affect existing parity tests (must remain at `max_abs_delta = 0` for Canary).
- NO remote push.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_jaxtoprow` on branch `tester/opus/m6b0r-jax-top-row-fix-followup`.

Write-only:
- `src/gpuwrf/dynamics/acoustic_wrf.py` — 3-line split of `top_denom`
- `tests/test_m6b0r_jax_top_row_synthetic.py` (NEW) — synthetic `c1f[nz-1] != c1f[nz]` test
- `.agent/sprints/2026-05-25-m6b0r-jax-top-row-fix-followup/` — proofs + memo

## Acceptance Criteria

### Stage 1 — Apply the fix (MANDATORY)

Split `top_denom` into `top_denom_a` and `top_denom_b`. Cite WRF source `:624-628` (a row) and `:644-648` (b row) for the index difference.

### Stage 2 — Synthetic regression test (MANDATORY)

Build a tiny synthetic state where `c1f[nz-1] = 0.5, c1f[nz] = 0.0` (non-zero transition at top). Assert JAX top-row `a` matches the WRF formula `(c1f[nz-1]*mut + c2f[nz-1]) * denominator` per source.

### Stage 3 — Re-run Canary parity (MANDATORY)

`python scripts/m6b0r_jax_vs_wrf_compare.py --operator calc_coef_w --tier all` — must still report `max_abs_delta = 0` on a/alpha/gamma (since Canary's `c1f[nz-1] = c1f[nz] = 0` makes the fix numerically inert).

### Stage 4 — No regression

```bash
pytest tests/test_m6b0r_jax_top_row_synthetic.py tests/test_m6b0r_calc_coef_w_fix.py tests/test_m6b0r_extractor_top_row.py tests/test_m6b0_*.py tests/test_m6b0r_*.py -v
```

Time budget: 30-60 min (3-line fix + small test).
