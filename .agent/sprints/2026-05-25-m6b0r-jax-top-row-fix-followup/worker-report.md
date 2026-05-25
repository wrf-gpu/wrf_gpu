# Worker Report — M6B0-R JAX Top-Row Fix Follow-up

## Objective

Mirror on the JAX side the extractor-fix sprint's split of the top-row
``calc_coef_w`` denominator so the ``a`` row (WRF ``module_small_step_em.F:626``)
uses ``c1f[nz-1]`` while the ``b`` row (WRF ``:646``) uses ``c1f[nz]``. The
Canary M6B0-R dataset has ``c1f[nz-1] = c1f[nz] = 0`` so the shared
``top_denom`` was silently inert; under any namelist where the top-face
hybrid-coordinate transition is non-trivial, the previous JAX code would
have produced the wrong top ``a`` coefficient.

## Files changed

- ``src/gpuwrf/dynamics/acoustic_wrf.py`` — split ``top_denom`` into
  ``top_denom_a = mass_h[nz-1] * mass_f[nz-1]`` (used by ``a[nz]``, WRF :626)
  and ``top_denom_b = mass_h[nz-1] * mass_f[nz]`` (used by ``b_top``, WRF :646),
  with an inline citation comment referencing the two WRF source ranges.
- ``tests/test_m6b0r_jax_top_row_synthetic.py`` (new) — three pytest cases
  on a synthetic ``DycoreMetrics`` where ``c1f[nz-1] = 0.5`` and
  ``c1f[nz] = 0.0`` (and distinct ``c2f`` at the top two faces): one asserts
  ``a[nz]`` matches the WRF :626 formula and that the buggy ``c1f[nz]``
  denominator would have produced a different answer; one infers ``b_top``
  from ``alpha[nz]`` with ``top_lid=True`` and asserts it matches WRF :646;
  one direct sensitivity guard confirming the two denominators differ.
- ``.agent/sprints/2026-05-25-m6b0r-jax-top-row-fix-followup/artifacts/proof_canary_parity_after_jax_top_row_fix.json``
  — comparator output across all three tiers post-fix.

## Commands run

- ``taskset -c 0-3 python -m pytest tests/test_m6b0r_jax_top_row_synthetic.py -v``
  → 3 passed.
- Bug-injection sanity check (manual sed swap of the ``a``-row denominator,
  reverted): ``test_top_a_row_uses_c1f_at_nz_minus_one`` failed with a 22×
  relative difference at the top face, confirming the test catches the bug.
- ``taskset -c 0-3 python scripts/m6b0r_jax_vs_wrf_compare.py --operator calc_coef_w --tier all --savepoint-root /tmp/wrf_gpu2_m6b0r/.agent/sprints/2026-05-24-m6b0r-real-fortran-emission/savepoints``
  → ``"passed": true, "outcome": "PASS"``, ``max_abs_delta = 0`` on
  ``a/alpha/gamma`` across column/patch16/golden × steps 1/2/5/10 (12
  savepoints × 3 fields = 36 zero-delta comparisons). Fix is numerically
  inert on Canary as expected because ``c1f[nz-1] = c1f[nz] = 0``.
- ``taskset -c 0-3 python -m pytest tests/test_m6b0r_jax_top_row_synthetic.py tests/test_m6b0r_calc_coef_w_fix.py tests/test_m6b0r_extractor_top_row.py tests/test_m6b0_*.py tests/test_m6b0r_*.py -v``
  → 24 passed.

## Proof objects

- ``.agent/sprints/2026-05-25-m6b0r-jax-top-row-fix-followup/artifacts/proof_canary_parity_after_jax_top_row_fix.json``
  — comparator JSON, ``"outcome": "PASS"`` with ``max_abs_delta = 0`` on
  every field/tier/step.
- ``tests/test_m6b0r_jax_top_row_synthetic.py`` — pinned regression that
  observably distinguishes the two denominators (bug-injection check
  produced a 22× relative error, demonstrating the test is non-vacuous).

## Unresolved risks

- The fix is observationally null on Canary; the synthetic test is the only
  positive evidence the JAX side now matches WRF :626 vs :646 indexing.
  When the next sprint extracts savepoints under a namelist with non-trivial
  top-face ``c1f`` transitions, the Canary-zero coincidence will lift and
  the JAX-vs-WRF comparator becomes the operational regression for this
  fix. No code change required at that point.

## Next decision

None — sprint is closed. Suggest hand-back to manager so the next M6B
extraction sprint (different vertical-coordinate transition) inherits the
fix without surprise.

---

## AGENT REPORT

Applied the 3-line JAX-side mirror of the extractor top-row fix in
``src/gpuwrf/dynamics/acoustic_wrf.py`` (split shared ``top_denom`` into
``top_denom_a`` using ``mass_f[nz-1]`` per WRF ``module_small_step_em.F:626``
and ``top_denom_b`` using ``mass_f[nz]`` per WRF :646) with an inline
citation comment. Added ``tests/test_m6b0r_jax_top_row_synthetic.py``
(3 cases) on a synthetic ``DycoreMetrics`` where ``c1f[nz-1] = 0.5`` and
``c1f[nz] = 0.0`` so the two denominators are arithmetically distinguishable;
a bug-injection sanity check (manual revert of the ``a``-row denominator)
produced a 22× relative error at ``a[nz]``, confirming the test is
non-vacuous. Re-ran the Canary comparator across all three tiers and the
fix remains numerically inert (``max_abs_delta = 0`` on ``a/alpha/gamma``
across 12 savepoints, ``"outcome": "PASS"``) because Canary's
``c1f[nz-1] = c1f[nz] = 0`` makes the index split silent — exactly the
prediction in the sprint contract. Full regression (synthetic + existing
calc_coef_w fix + extractor top-row + all M6B0/M6B0R tests) is 24/24 green.
