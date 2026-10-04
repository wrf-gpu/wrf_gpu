# v0.20 fp32 CPU Prototype Report

## Objective

CPU-only math de-risk for the v0.20 fp32 milestone: prove the naive fp32-total
form reproduces the known cancellation failure, while the
perturbation-authoritative form with fp64 base gradients, local fp64 brackets,
fp64 tridiagonal solve, and compensated accumulation avoids it on idealized
sea/lee-wave/steep-ridge/high-peak columns.

No `src/gpuwrf` files were touched.

## Outcome

PASS for the CPU prototype scope.

Key numbers from `proof_results.json`:

| Metric | Result |
|---|---:|
| Cases | 4 |
| Local cancellation gate threshold | 27.0x |
| Naive cases failing local cancellation gate | 4 / 4 |
| Min naive/perturbation cancellation error ratio | 34.8329x |
| Median naive/perturbation cancellation error ratio | 470.328x |
| Max naive/perturbation cancellation error ratio | 9292.75x |
| Perturbation cases passing bands | 4 / 4 |
| Accumulator drift reduction | 2.17e10x |

Per-case highlights:

| Case | p ratio | ph ratio | mu ratio | Perturb W RMSE | Perturb P RMSE | MU rel drift |
|---|---:|---:|---:|---:|---:|---:|
| sea | 175.241x | 1001.716x | 9292.755x | 1.829e-4 | 2.733e-3 | 4.214e-12 |
| lee_wave | 91.280x | 535.515x | 9148.837x | 2.074e-4 | 4.183e-3 | 1.026e-12 |
| steep_ridge | 70.142x | 405.140x | 3604.954x | 2.467e-4 | 6.236e-3 | 8.557e-12 |
| high_peak | 34.833x | 351.731x | 2883.224x | 8.828e-4 | 1.236e-2 | 1.414e-11 |

Accumulator drift experiment:

- fp64 truth final: `100504.06849607742`
- naive fp32 abs error: `83.15806642257667`
- Kahan fp32 abs error: `3.827153705060482e-09`
- naive/Kahan error ratio: `21728436543.48669`

## Files Changed

New files only under `proofs/v020/fp32_proto/`:

- `__init__.py`
- `acceptance_bands.py`
- `acceptance_bands_snapshot.json`
- `fp32_column_proto.py`
- `proof_results.json`
- `pytest.log`
- `run_proof.py`
- `test_fp32_proto.py`
- `REPORT.md`

## Commands Run

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python -m pytest proofs/v020/fp32_proto -q
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python proofs/v020/fp32_proto/run_proof.py --nsteps 20000
```

Pytest result: `4 passed in 2.94s`.

## Proof Objects

- `proof_results.json`: full 20,000-substep CPU prototype results.
- `acceptance_bands_snapshot.json`: importable v0.20 band table snapshot from
  `FINAL_FP32_SPRINT_PLAN.md` section 1 and `GPT_FP32_PLAN.md` section 7.
- `pytest.log`: unit-test proof for acceptance bands, local cancellation failure,
  perturbation long-substep stability, and compensated accumulation.

## Honest Limits / Risks

- This is a synthetic analytic proof, not a WRF fixture/savepoint and not a
  production dycore validation.
- The naive total-field path fails the local cancellation gate strongly, but in
  this damped synthetic long-run it still stays inside the relaxed forecast-style
  W/P/MU bands. The failing baseline here is the v0.17-style local cancellation
  mechanism, not a claim that every damped toy forecast blows up.
- No perturbation-form struggle appeared in this prototype. The largest
  perturbation W RMSE was `8.828e-4`, far below the 24 h wind band of `0.25`.
- Production risk remains HLO/liveness and WRF algebra placement: the next
  proof must use WRF savepoints/fixtures and verify no hot-loop fp32 total
  differencing or broad fp64 promotion.

## Next Decision Needed

No human decision is needed for this CPU proof. The dycore owner can use this as
supporting evidence for implementing R1/R3 in production, gated by WRF fixture
oracles and HLO liveness audits.

## Addendum: Real WRF Column Validation

Source: `.agent/sprints/2026-05-25-m6-perf-design-acceptance/artifacts/wrfout_d02_1h_cpu_reference.nc`.
This full CPU-reference wrfout contains `P/PB/PH/PHB/MU/MUB/W`, so no fallback
savepoint was needed. The proof extracted three interior real WRF columns:
highest terrain, steepest east-west terrain gradient, and lowest terrain.

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python proofs/v020/fp32_proto/realcolumn_wrfout_proof.py
```

Proof object: `proof_results_realcolumn.json`.

Outcome: PASS for the real-column CPU prototype scope.

| Metric | Result |
|---|---:|
| Real columns | 3 |
| Prototype horizon | 1000 substeps |
| Prototype dt | 0.05 |
| Local cancellation gate threshold | 27.0x |
| Naive cases failing local cancellation gate | 3 / 3 |
| Perturbation cases finite | 3 / 3 |
| Perturbation cases passing `acceptance_bands.py` checks | 3 / 3 |
| Max naive/perturbation cancellation error ratio | 3.109e27x |

Per-column perturbation-vs-fp64-reference metrics:

| Column | Terrain m | p ratio | ph ratio | mu ratio | W RMSE | P RMSE | PH RMSE | MU rel drift | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| high terrain | 2987.118 | 2.086e27x | 16.407x | 1.000x | 2.138e-3 | 1.493e-3 | 7.019e-2 | 6.294e-9 | yes |
| steep gradient | 2101.070 | 1.818e27x | 15.955x | 27.000x | 5.123e-2 | 4.335e-2 | 1.655e0 | 5.755e-8 | yes |
| low terrain | 0.000 | 3.109e27x | 16.571x | 27.000x | 2.315e-5 | 1.039e-3 | 3.239e-3 | 3.864e-9 | yes |

Interpretation:

- The real wrfout columns reproduce the core cancellation mechanism on real WRF
  profiles: naive fp32 totals lose the small `P` face signal in all three
  columns; the perturbation form does not.
- The perturbation path tracks the fp64 prototype reference inside the imported
  v0.20 bands for all three real columns and stays finite.
- The earlier synthetic-only caveat is narrowed: the math now has a real WRF
  column proof. Remaining caveat: this still validates the CPU NumPy prototype
  against its fp64 reference initialized from real WRF profiles. It is not yet a
  production WRF operator/savepoint parity proof for `src/gpuwrf`.
