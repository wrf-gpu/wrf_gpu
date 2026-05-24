# Worker Report — M6B0-R Defect Analysis `calc_coef_w`

## objective

Find the WRF/JAX formulation gap for `calc_coef_w`, apply a minimal WRF-cited fix, and prove `a`, `alpha`, and `gamma` parity on the existing M6B0-R column, patch16, and golden savepoints.

## defect mechanism

The previous JAX comparison path used `build_epssm_column_coefficients(theta, dz_m)`, a MPAS-family/geometric-meter coefficient builder. WRF `calc_coef_w` does not use `theta` or `dz_m` for these coefficients. It builds `a`, `alpha`, and `gamma` from eta-coordinate `rdn/rdnw`, column mass `MUT`, hybrid pressure-mass factors `c1h/c2h` and `c1f/c2f`, and then runs the Thomas forward recurrence. Source: `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/WRF/dyn_em/module_small_step_em.F:624-649`.

The main gap was formulation/unit/staggering: meter-space `1/dz` second-derivative coefficients were being compared against eta-pressure hybrid coefficients. This also populated the WRF lower boundary-adjacent row incorrectly; WRF explicitly sets `a(i,2,j)=0` before the interior `kk=3..kde-1` loop. The source run has `TOP_LID=F`, so line 620 leaves `lid_flag=1`; the savepoint metadata field saying `top_lid=True` is stale relative to the actual namelist and stored top-row coefficients.

## files changed

- `src/gpuwrf/dynamics/acoustic_wrf.py`: added `calc_coef_w_wrf_coefficients`, matching WRF lines 624-649 with explicit hybrid denominators and forward `alpha/gamma` recurrence.
- `scripts/m6b0r_jax_vs_wrf_compare.py`: changed the M6B0-R comparator to call the WRF-shaped JAX coefficient helper with savepoint `mut` and WRF source metrics.
- `tests/test_m6b0r_calc_coef_w_fix.py`: added regression coverage for column, patch16, and golden tiers, plus an improvement assertion against the legacy `theta/dz_m` builder.
- `.agent/sprints/2026-05-24-m6b0r-defect-analysis-calc-coef-w/defect_analysis.md`: forensic line table and discrepancy report.
- `.agent/sprints/2026-05-24-m6b0r-defect-analysis-calc-coef-w/proof_calc_coef_w_before.{txt,json}`
- `.agent/sprints/2026-05-24-m6b0r-defect-analysis-calc-coef-w/proof_calc_coef_w_after.{txt,json}`
- `.agent/sprints/2026-05-24-m6b0r-defect-analysis-calc-coef-w/proof_calc_coef_w_delta.txt`
- `.agent/sprints/2026-05-24-m6b0r-defect-analysis-calc-coef-w/proof_no_regression.txt`

## commands run

- `python scripts/m6b0r_jax_vs_wrf_compare.py --operator calc_coef_w --tier all --savepoint-root /tmp/wrf_gpu2_m6b0r/.agent/sprints/2026-05-24-m6b0r-real-fortran-emission/savepoints --output .agent/sprints/2026-05-24-m6b0r-defect-analysis-calc-coef-w/proof_calc_coef_w_before.json > .agent/sprints/2026-05-24-m6b0r-defect-analysis-calc-coef-w/proof_calc_coef_w_before.txt`
- `XLA_FLAGS=--xla_force_host_platform_device_count=4 OMP_NUM_THREADS=4 python scripts/m6b0r_jax_vs_wrf_compare.py --operator calc_coef_w --tier all 2>&1 | tee .agent/sprints/2026-05-24-m6b0r-defect-analysis-calc-coef-w/proof_calc_coef_w_after.txt`
- `XLA_FLAGS=--xla_force_host_platform_device_count=4 OMP_NUM_THREADS=4 python scripts/m6b0r_jax_vs_wrf_compare.py --operator calc_coef_w --tier all --output .agent/sprints/2026-05-24-m6b0r-defect-analysis-calc-coef-w/proof_calc_coef_w_after.json`
- `XLA_FLAGS=--xla_force_host_platform_device_count=4 OMP_NUM_THREADS=4 pytest tests/test_m6b0r_calc_coef_w_fix.py -v`
- `XLA_FLAGS=--xla_force_host_platform_device_count=4 OMP_NUM_THREADS=4 pytest tests/test_m6x_*.py tests/test_m3_transfer_audit.py tests/test_m6b0_*.py tests/test_m6b0r_*.py tests/test_m6b0r_calc_coef_w_fix.py -v 2>&1 | tee .agent/sprints/2026-05-24-m6b0r-defect-analysis-calc-coef-w/proof_no_regression.txt`

## proof objects produced

- `proof_calc_coef_w_before.json`: `PARITY-DEFECT-LOCALIZED`
- `proof_calc_coef_w_after.json`: `PASS`
- `proof_calc_coef_w_delta.txt`: worst deltas improved to zero on all tiers
- `proof_no_regression.txt`: `88 passed in 303.94s`

Before/after worst deltas:

| tier | a | alpha | gamma |
|---|---:|---:|---:|
| column | `259.66394743107992 -> 0` | `0.99500258576967182 -> 0` | `0.47943800307554663 -> 0` |
| patch16 | `278.0947834046479 -> 0` | `0.99500838299690852 -> 0` | `0.47968748791035459 -> 0` |
| golden | `266.29365776579152 -> 0` | `0.99501004033830887 -> 0` | `0.47986483643350919 -> 0` |

## unresolved risks

- This fix proves the WRF-shaped coefficient construction against the M6B0-R savepoints. The older production `_calc_coef_w` path still exists for ADR-023 runtime behavior; replacing that runtime path with WRF forward-sweep coefficients is a separate interface decision because current solver callers expect raw tridiagonal `a,b,c`, not WRF's `a,alpha,gamma` forward-sweep products.
- The M6B0-R savepoints still carry stale `top_lid=True` metadata, but their coefficients and source namelist match `TOP_LID=F`. The comparator documents and uses the source-run branch.

## next decision needed

Outcome: `FIRST-OPERATOR-PARITY-ACHIEVED` for the M6B0-R coefficient proof. Manager can dispatch M6B1 (`advance_mu_t` parity), with a separate decision later on whether to replace the ADR-023 runtime tridiagonal interface with a WRF forward/back-sweep interface.
