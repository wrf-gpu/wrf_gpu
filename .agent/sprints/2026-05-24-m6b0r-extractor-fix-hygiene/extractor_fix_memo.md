# M6B0-R Extractor Top-Row Hygiene — Tester Memo

Sprint contract: `.agent/sprints/2026-05-24-m6b0r-extractor-fix-hygiene/sprint-contract.md`
Branch: `tester/opus/m6b0r-extractor-fix-hygiene`
Worktree: `/tmp/wrf_gpu2_extfix`

## Bugs Fixed

### Bug 1 — `lid_flag` hardcoded to `1.0` (extractor:138)

WRF Fortran source (`/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/dyn_em/module_small_step_em.F:619-620`):

```fortran
lid_flag=1
IF(top_lid)lid_flag=0
```

Old Python (`scripts/m6b0r_wrf_savepoint_extract.py:138`):

```python
lid_flag = 1.0   # ignored state["attrs"]["top_lid"]
```

Fix: `_wrf_calc_coef_w` now takes an optional `top_lid` kwarg and otherwise
reads `state["attrs"]["top_lid"]`. The new `_resolve_top_lid` helper picks
the value with explicit precedence: (1) wrfout `TOP_LID` attribute (bool or
Fortran `T/F` string), (2) sibling `namelist.output`, (3) sibling
`namelist.input`, (4) WRF default `.false.`. The Canary d02 wrfout omits
the attribute but `namelist.output` carries `TOP_LID= 11*F`, so the resolver
returns `False` and `lid_flag` stays `1.0` for the current run — matching the
operational WRF behaviour described in the prior worker report
(`.agent/sprints/2026-05-24-m6b0r-defect-analysis-calc-coef-w/worker-report.md:11`).

### Bug 2 — Top-row `denom_top` reused for both `a` and `b` (extractor:142)

WRF Fortran source `module_small_step_em.F:626` (top `a` row, with `k=kde-1`):

```fortran
a(i,kde,j) = -2.*cof*rdnw(kde-1)**2*c2a(i,kde-1,j)*lid_flag &
             /((c1h(k)*MUT(i,j)+c2h(k))*(c1f(k)*MUT(i,j)+c2f(k)))
```

and `module_small_step_em.F:646` (top `b` row, with `k=kde`):

```fortran
b = 1.+2.*cof*rdnw(kde-1)**2*c2a(i,kde-1,j) &
       /((c1h(k-1)*MUT(i,j)+c2h(k-1))*(c1f(k)*MUT(i,j)+c2f(k)))
```

The two top denominators differ in the `c1f`/`c2f` index: top-`a` uses
`c1f(kde-1)`, top-`b` uses `c1f(kde)`. The old Python collapsed them into a
single `denom_top` that used `c1f[nz]` (i.e., `c1f(kde)`) — correct for the
`b_top` line but wrong for the top `a` line.

Fix: extractor now computes two explicit denominators —
`denom_top_a = (c1h[nz-1]*mut + c2h[nz-1]) * (c1f[nz-1]*mut + c2f[nz-1])`
and `denom_top_b = (c1h[nz-1]*mut + c2h[nz-1]) * (c1f[nz]*mut + c2f[nz])`
— used at the top `a` and top `b` rows respectively.

## Regression Test (`tests/test_m6b0r_extractor_top_row.py`)

11 tests across three classes:

| Class | Test | What it pins |
|---|---|---|
| `TestLidFlag` | `test_top_lid_true_zeros_a_top_row` | `top_lid=True -> lid_flag=0 -> a[nz]=0` |
| `TestLidFlag` | `test_top_lid_false_produces_canonical_a_top_row` | `top_lid=False -> lid_flag=1` and matches WRF formula bit-for-bit |
| `TestLidFlag` | `test_explicit_kwarg_overrides_state_attrs` | kwarg precedence over `state["attrs"]` |
| `TestTopRowDenomIndex` | `test_top_a_row_uses_c1f_at_kde_minus_one` | top `a` denom = `c1f[nz-1]` not `c1f[nz]` |
| `TestTopRowDenomIndex` | `test_top_a_row_with_wrong_c1f_index_would_differ` | synthetic fixture is index-sensitive (no false negative) |
| `TestTopRowDenomIndex` | `test_top_b_row_uses_c1f_at_kde` | top `b` denom (via `1/alpha[nz]` with `a[nz]=0`) = `c1f[nz]` |
| `TestResolveTopLid` | 5 variants | wrfout bool/string attr, namelist fallback returns `False` for Canary d02 |

The fixture in `_synthetic_state` uses monotone `c1f` so that `c1f[nz-1]=0.74`
and `c1f[nz]=0.8` produce numerically distinguishable top-row denominators —
this would have caught both bugs deterministically.

Proof: `proof_new_test.txt` — 11 passed in 0.06s.

## Parity After Fix

Comparator: `python scripts/m6b0r_jax_vs_wrf_compare.py --operator calc_coef_w --tier all`

Result: `PASS` on all three tiers (column, patch16, golden), all four
substeps (1, 2, 5, 10), all three fields (`a`, `alpha`, `gamma`).
Every `max_abs_delta` = `0.0`. Outcome JSON top-level `"outcome": "PASS"`.

Proof artifacts:
- `proof_calc_coef_w_after_extractor_fix.txt` — full pytest-style transcript
- `proof_calc_coef_w_after_extractor_fix.json` — structured payload

### Why parity didn't regress (key finding)

For the Canary d02 hybrid-eta column the top two faces have
`c1f[nz-1] = c1f[nz] = 0.0` and `c2f[nz-1] = c2f[nz] = 95000.0` (i.e., the
hybrid-coordinate transition has completed below the model top — the top
two faces are pure pressure-coordinate). Therefore:

1. The wrong `c1f[nz]` index in the old top-`a` denominator was
   numerically equal to the correct `c1f[nz-1]` value for this dataset.
2. The hardcoded `lid_flag=1.0` was correct because the Canary d02
   namelist sets `TOP_LID=F`, so canonical `lid_flag=1` anyway.

Side-by-side bit comparison of the regenerated savepoints against the
pre-fix savepoints living at `/tmp/wrf_gpu2_m6b0r/.../savepoints/` shows
`max_abs_delta = 0.0` for `a`, `alpha`, and `gamma` on every tier
(`proof_savepoint_delta_old_vs_new.txt`).

The JAX side (`src/gpuwrf/dynamics/acoustic_wrf.py:635`) still uses the
same single `top_denom = mass_h[nz-1] * mass_f[nz]` for both top rows
— i.e., it carries the same latent Bug 2 pattern as the old extractor.
That code path is OUT-OF-SCOPE for this sprint per the contract's
non-goals. It produces the right numbers only because of the
hybrid-coordinate accident above. **Recommendation**: open a follow-up
M6B0-R-JAX-TOP-ROW-FIX sprint to split `top_denom_a`/`top_denom_b` in
the JAX helper (3-line localized fix), with the new extractor regression
test as the parity oracle.

## Files Changed

- `scripts/m6b0r_wrf_savepoint_extract.py` — added `_resolve_top_lid` helper;
  `_wrf_calc_coef_w` now reads `top_lid` from state/kwarg and computes
  explicit `denom_top_a`/`denom_top_b` per WRF source.
- `tests/test_m6b0r_extractor_top_row.py` (NEW) — 11 regression tests.
- `.agent/sprints/2026-05-24-m6b0r-real-fortran-emission/savepoints/{column,patch16,golden}/`
  — savepoints regenerated with corrected extractor (bit-identical for this
  dataset).
- `.agent/sprints/2026-05-24-m6b0r-extractor-fix-hygiene/` —
  `extractor_fix_memo.md`, `proof_new_test.txt`,
  `proof_calc_coef_w_after_extractor_fix.{txt,json}`,
  `proof_no_regression.txt`,
  `proof_savepoint_delta_old_vs_new.txt`.

## Commands Run

```bash
taskset -c 0-3 pytest tests/test_m6b0r_extractor_top_row.py -v  # 11 passed
taskset -c 0-3 python scripts/m6b0r_wrf_savepoint_extract.py --tier column  --steps 10
taskset -c 0-3 python scripts/m6b0r_wrf_savepoint_extract.py --tier patch16 --steps 10
taskset -c 0-3 python scripts/m6b0r_wrf_savepoint_extract.py --tier golden  --steps 10
taskset -c 0-3 python scripts/m6b0r_jax_vs_wrf_compare.py --operator calc_coef_w --tier all  # PASS
taskset -c 0-3 pytest tests/test_m6x_*.py tests/test_m3_transfer_audit.py tests/test_m6b0_*.py \
    tests/test_m6b0r_*.py tests/test_m6b0r_calc_coef_w_fix.py \
    tests/test_m6b0r_extractor_top_row.py -v
```

## No Regression

Full suite per contract Stage 5: `92 passed, 2 skipped, 3 failed, 2 errored`.
The 5 failures+errors are pre-existing GPU-only tests
(`test_m6x_pressure_diagnose_wiring`, `test_m6x_warm_bubble_operator_sanity`,
`test_m6x_tier3_convergence_infra::test_runner_smoke_produces_contract_schema`)
that call `State.zeros(grid)` which requires a GPU JAX backend; this CPU-only
worker honours `taskset -c 0-3 JAX_PLATFORM_NAME=cpu` per the constitution and
cannot exercise them. None of those tests reference `m6b0r`, the extractor,
or any code path touched by this sprint; the prior worker (M6B0-R defect
analysis, commit `ac252e8`) reported 88/88 PASS on a GPU-enabled session, so
these are environmental, not regressions.

Targeted subset passes 25/25 with no skips:

```
pytest tests/test_m6b0r_calc_coef_w_fix.py tests/test_m6b0_*.py tests/test_m6b0r_*.py \
       tests/test_m3_transfer_audit.py tests/test_m6b0r_extractor_top_row.py
# 25 passed in 8.01s
```

## Unresolved Risks

- The JAX helper `calc_coef_w_wrf_coefficients`
  (`src/gpuwrf/dynamics/acoustic_wrf.py:635`) carries the same `top_denom`
  pattern (Bug 2). It is latent for the same hybrid-coordinate reason but
  should be split for canonical correctness. Recommended follow-up sprint:
  M6B0-R-JAX-TOP-ROW-FIX, 3-line patch in `acoustic_wrf.py:635-637,660`
  with the new extractor regression test as the gating proof.
- The savepoint metadata `namelist_hash` SHA-256 may shift slightly across
  re-extracts even though arrays are bit-identical, because the resolver now
  reads `top_lid=False` (was `True`) on this dataset. This is the correct
  behaviour (matches operational WRF) but downstream consumers that compare
  manifests across sprints should expect a one-time hash change.

## AGENT REPORT

**GO** for M6B1. Both extractor top-row bugs in
`scripts/m6b0r_wrf_savepoint_extract.py:138,142` are now canonical:
`lid_flag` is read from a 4-way precedence chain (wrfout attr ->
namelist.output -> namelist.input -> WRF default `False`) and the top `a`
and `b` rows now use explicit `denom_top_a` (with `c1f[nz-1]`) and
`denom_top_b` (with `c1f[nz]`) per WRF
`module_small_step_em.F:619-620,626,646`. The 11-test regression
`tests/test_m6b0r_extractor_top_row.py` would fail if either bug returned.
The `calc_coef_w` comparator still reports `PASS` on all three tiers
(`a/alpha/gamma` worst delta = `0.0`) because the Canary d02 column has
`c1f[nz-1] = c1f[nz] = 0.0` at the top, so the fix is numerically inert
for this dataset while structurally correct. Regenerated savepoints are
bit-identical to the pre-fix ones. The JAX side
(`acoustic_wrf.py:calc_coef_w_wrf_coefficients`) carries the same latent
Bug 2 pattern; flagged as an out-of-scope follow-up
(`M6B0-R-JAX-TOP-ROW-FIX`, 3-line patch) but does NOT block M6B1.
