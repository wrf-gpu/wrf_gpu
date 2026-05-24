# Audit Memo — M6B0-R Reproducer Audit (opus tester)

Sprint: `2026-05-24-m6b0r-reproducer-audit`
Branch: `tester/opus/m6b0r-reproducer-audit`
Worktree: `/tmp/wrf_gpu2_reprod`
Date: 2026-05-24

## 1) Is M6B0-R's `PARITY-DEFECT-LOCALIZED` verdict reproducible?

**YES — bitwise identical reproduction.**

Re-ran `scripts/m6b0r_jax_vs_wrf_compare.py --operator calc_coef_w --tier all` unchanged
against the original M6B0-R savepoints under `/tmp/wrf_gpu2_m6b0r/.agent/sprints/2026-05-24-m6b0r-real-fortran-emission/savepoints/`.

Worst-case deltas matched M6B0-R's reported values exactly:

| Tier   | a (max_abs)         | alpha (max_abs)        | gamma (max_abs)        |
|--------|---------------------|------------------------|------------------------|
| column | 259.6639474310799   | 0.9950025857696718     | 0.4794380030755466     |
| patch16| 278.0947834046479   | 0.9950083829969085     | 0.4796874879103546     |
| golden | 266.2936577657915   | 0.9950100403383089     | 0.4798648364335092     |

All three tiers FAIL the `1e-11` absolute tolerance ladder. Outcome: `PARITY-DEFECT-LOCALIZED`.
Proof: `proof_reproduce_stage5.txt` / `proof_reproduce_stage5.json`.

## 2) Is the M6B0-R Python `_wrf_calc_coef_w` reproduction faithful to WRF Fortran `:570-652`?

**PARTIAL** — body is faithful; two localized top-row bugs do NOT explain the divergence.

Full audit: `proof_python_reproduction_audit.md`. Summary:

- Interior `a` loop (Fortran 629-633), interior `b/c/alpha/gamma` recurrence (Fortran 635-643),
  and `b_top` formula (Fortran 644-650) all translate correctly with proper 1-indexed-to-0-indexed
  arithmetic. Verified each index by hand.
- **Bug #1**: `lid_flag` hardcoded to `1.0` at `scripts/m6b0r_wrf_savepoint_extract.py:138`.
  WRF spec: `lid_flag = 0 if top_lid else 1`. With `top_lid=True` (extractor default since wrfout
  lacks the attribute) WRF would zero `a(kde)`; Python computes a non-zero value.
- **Bug #2**: `denom_top` at line 142 uses `c1f[nz]` (Fortran `c1f(kde)`) instead of
  `c1f[nz-1]` (Fortran `c1f(kde-1)`). Only matters for `a[nz]`; the SAME `denom_top` is reused
  for `b_top` where the index IS correct, so the bug only affects the top `a` row.
- **Design caveats** (worker report Risk #2): `cqw` and `c2a` are placeholder `ones`, not WRF
  state. These are identical on both sides of the JAX-vs-extractor compare, so they cancel and
  do not contribute to the parity gap; they DO limit what this savepoint suite proves about
  full-state WRF parity.

Bugs #1+#2 combined produce only:
- `a[44]` independent-vs-extractor delta = **1.03e-2**
- `alpha[44]` delta = **4.95e-5**

These live at index 44 (top W-face) only. The PARITY-DEFECT-LOCALIZED max deltas live at
**k=2** (a, alpha) and **k=40** (gamma). The bugs do NOT explain the JAX gap.

## 3) Independent recomputation — who is "right" for a/alpha/gamma?

Script: `independent_recompute.py` (worktree only, NOT committed to `scripts/`).
Proof: `proof_independent_recomputation.json`.

Computed `(a, alpha, gamma)` for the center column using a fresh, strict translation of
WRF Fortran `:570-652` written from the source (not derived from the M6B0-R reproduction).
Used identical `cqw=ones, c2a=ones` placeholders so any deltas isolate to the `calc_coef_w`
arithmetic itself.

| Comparison | a max_abs | alpha max_abs | gamma max_abs |
|------------|-----------|---------------|---------------|
| independent vs M6B0-R extractor in-mem | 1.03e-2 (at k=44) | 4.95e-5 (at k=44) | 0.0 |
| independent vs committed savepoint     | 1.03e-2 (at k=44) | 4.95e-5 (at k=44) | 0.0 |
| **independent vs JAX**                 | **259.66 (at k=2)** | **0.995 (at k=2)** | **0.479 (at k=40)** |
| M6B0-R extractor vs JAX                | 259.66 (at k=2)   | 0.995 (at k=2)| 0.479 (at k=40) |

Sanity: M6B0-R in-memory recomputation matches the committed HDF5 savepoint values bitwise (0.0 delta).

**Verdict**: my independent reading agrees with the M6B0-R reproduction on the interior to within
the two known top-row bugs. **JAX diverges from both Python readings of WRF by the same large
magnitude at the same interior cells.** The defect is in the JAX side (or, more precisely, in
the comparator's choice of mapping the JAX raw tridiagonal `(tri_a, tri_b, tri_c)` to
`(a, alpha, gamma)` via `alpha=1/tri_b, gamma=tri_c/tri_b`).

### A meta-concern flagged for DEFECT-ANALYSIS lane

The JAX `build_epssm_column_coefficients` does NOT compute the WRF `(a, alpha, gamma)` objects.
It computes raw tridiagonal coefficients `(tri_a, tri_b, tri_c)` from a different formulation
(MPAS-A `cofrz/cofwr/cofwz/coftz/cofwt`, using only `theta, dz_m, dt, epssm` — no
`mut, c1h/c2h/c1f/c2f, rdn, rdnw, cqw, c2a`). The comparator at
`scripts/m6b0r_jax_vs_wrf_compare.py:46-50` coerces:
```
a     <-> tri_a
alpha <-> 1.0 / tri_b
gamma <-> tri_c / tri_b
```

But WRF's `alpha` and `gamma` are **Thomas-elimination workspace variables** defined by the
recurrence `alpha(k) = 1/(b - a(k)*gamma(k-1))` and `gamma(k) = c*alpha(k)`. They are NOT
`1/b` and `c/b`. The comparator's mapping is mathematically wrong even for an exact reimplementation of WRF — there is no choice of `tri_a, tri_b, tri_c` for which `1/tri_b` equals WRF's `alpha` except in the trivial case `a(k)*gamma(k-1)=0` (which holds only at the bottom row).

This is a structural mismatch in the validation harness, on top of the underlying formulation
difference between MPAS-A and WRF. DEFECT-ANALYSIS should address both — the harness coercion
AND the JAX side's actual physical formulation — before declaring the defect "fixed".

## 4) GO / NO-GO for the DEFECT-ANALYSIS lane

**GO — with two caveats.**

GO rationale:
- The `PARITY-DEFECT-LOCALIZED` verdict is reproducible bitwise.
- The M6B0-R Python reproduction is faithful enough that its expected `(a, alpha, gamma)` are
  trustworthy for the INTERIOR levels where the JAX gap dominates (k=2..43).
- My independent strict-WRF reading agrees with the M6B0-R reproduction to within 1e-5 at the
  single top row, vs the JAX gap of O(1)..O(100) at interior cells. The defect is REAL and
  REPRODUCIBLE.

Caveats DEFECT-ANALYSIS must own before merging any fix:
1. **Top-row top_lid + c1f(kde) bugs in `_wrf_calc_coef_w`** (extractor lines 138 and 142):
   any fix that targets matching `a[44]` exactly is chasing a `_extractor_` artifact, not WRF.
   If DEFECT-ANALYSIS plans to bring `a[44]` to bitwise parity with the existing savepoints,
   it will lock in two extractor bugs as "the WRF answer". Either re-emit savepoints with the
   bug fixed (re-running M6B0-R extractor) or scope any "fix" claim to interior levels only.
2. **Harness-level formulation mismatch (`alpha=1/tri_b`, `gamma=tri_c/tri_b`)**: as documented
   in Part 3, the comparator's mapping does not align with WRF's Thomas-elimination definitions
   of `alpha` and `gamma`. Even a perfect WRF reimplementation in JAX would still fail this
   comparator. DEFECT-ANALYSIS must either change the JAX side to expose actual `(a, alpha, gamma)`
   (with the recurrence applied) or change the comparator to compare like-for-like raw
   tridiagonal entries. Both lanes should agree on the contract before code lands.

The verdict itself is trustworthy enough to act on. The proposed fix should not be evaluated
solely against the existing savepoint set; a re-emission after fixing the two extractor bugs is
recommended before claiming bitwise parity at the top row.

## 5) No-regression

`pytest --collect-only` → `585 tests collected in 1.74s`. No source files modified
(`git status` shows only additions under `.agent/sprints/2026-05-24-m6b0r-reproducer-audit/`).
Proof: `proof_no_touch.txt`.

## Artifacts

- `proof_reproduce_stage5.txt` / `.json` — Part 1
- `proof_python_reproduction_audit.md` — Part 2
- `independent_recompute.py` + `proof_independent_recomputation.json` — Part 3
- `audit_memo.md` (this file) — Part 4
- `proof_no_touch.txt` — Part 5

---

## AGENT REPORT

Verified M6B0-R's `PARITY-DEFECT-LOCALIZED` verdict bitwise-reproduces on all three tiers
(column/patch16/golden) by re-running the unchanged comparator against the original savepoints.
Line-by-line audit of the extractor's Python `_wrf_calc_coef_w` against WRF Fortran
`module_small_step_em.F:570-652` confirms the interior arithmetic is a faithful translation
but flags two localized top-row bugs: `lid_flag` is hardcoded to 1.0 instead of honoring
`top_lid`, and `denom_top` reuses `c1f[nz]` where Fortran wants `c1f(kde-1)`; together these
produce only 1e-2 in `a[44]` and 5e-5 in `alpha[44]`. My independent strict-WRF recomputation
agrees with the extractor to within those bugs on the interior, and JAX diverges from both
Python readings by the same O(1)..O(100) magnitudes at interior cells k=2 and k=40 — the
defect is REAL and not an artifact of the extractor. **Verdict: GO** for DEFECT-ANALYSIS to act,
with two caveats: (a) the two top-row extractor bugs must be acknowledged or savepoints
re-emitted before claiming bitwise parity at k=44, and (b) the comparator's structural mapping
`alpha=1/tri_b, gamma=tri_c/tri_b` does not match WRF's Thomas-recurrence definitions of
`alpha` and `gamma`, so the harness contract itself needs review.
