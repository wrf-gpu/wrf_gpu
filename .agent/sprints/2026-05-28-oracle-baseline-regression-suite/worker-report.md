# Worker Report - Oracle-Baseline Regression Suite

## Verdict

`ORACLE_SUITE_COMPLETE`

Headline: 2 field tests passing baseline today; 14 failing against predeclared tolerances in the 1h smoke; 3 cases blocked-pending-Oracle in the manifest.

## Objective

Build a multi-case CPU-WRF-Oracle regression suite with a case manifest, tolerance ladder, GPU-vs-Oracle driver, milestone snapshot/regression checker, pytest smoke entry, documentation, and proof objects.

## Files Changed

- `tests/regression/oracle_cases.yaml`
- `tests/regression/tolerances.yaml`
- `tests/regression/README.md`
- `tests/regression/test_regression_suite.py`
- `scripts/run_regression_suite.py`
- `scripts/run_milestone_regression.sh`
- `proofs/regression/**`
- `.agent/sprints/2026-05-28-oracle-baseline-regression-suite/smoke_output.json`
- `.agent/sprints/2026-05-28-oracle-baseline-regression-suite/smoke_output.force.json`
- `.agent/sprints/2026-05-28-oracle-baseline-regression-suite/worker-report.md`

No `src/gpuwrf/**`, M11/M12/M13 worker-owned files, governance files, or `/home/enric/src/wrf_gpu/` files were edited.

## Commands Run

- `taskset -c 0-3 python -m py_compile scripts/run_regression_suite.py tests/regression/test_regression_suite.py`
- `taskset -c 0-3 python scripts/run_regression_suite.py --smoke > .agent/sprints/2026-05-28-oracle-baseline-regression-suite/smoke_output.json`
- `taskset -c 0-3 python scripts/run_regression_suite.py --smoke --force-gpu-run > .agent/sprints/2026-05-28-oracle-baseline-regression-suite/smoke_output.force.json`
- `taskset -c 0-3 python scripts/run_regression_suite.py --smoke > .agent/sprints/2026-05-28-oracle-baseline-regression-suite/smoke_output.json`
- `taskset -c 0-3 pytest tests/regression/test_regression_suite.py`
- `taskset -c 0-3 pytest tests/savepoint/ tests/regression/`
- `taskset -c 0-3 bash -n scripts/run_milestone_regression.sh`
- `taskset -c 0-3 python scripts/run_regression_suite.py --smoke --milestone-snapshot SMOKE > proofs/regression/snapshot_SMOKE.command.json`
- `taskset -c 0-3 python scripts/run_regression_suite.py --compare-snapshot SMOKE > proofs/regression/regression_check_NONE_to_SMOKE.command.json`

## Proof Objects Produced

- `proofs/regression/canary_20260521_24h_d02_SMOKE.json`
- `proofs/regression/aggregate_SMOKE.json`
- `proofs/regression/gpu_runs/canary_20260521_24h_d02_SMOKE/pipeline_run_20260521.json`
- `proofs/regression/gpu_runs/canary_20260521_24h_d02_SMOKE/gpu_run.stdout.txt`
- `proofs/regression/gpu_runs/canary_20260521_24h_d02_SMOKE/gpu_run.stderr.txt`
- `proofs/regression/snapshot_SMOKE.json`
- `proofs/regression/regression_check_NONE_to_SMOKE.json`
- `.agent/sprints/2026-05-28-oracle-baseline-regression-suite/smoke_output.json`
- `.agent/sprints/2026-05-28-oracle-baseline-regression-suite/smoke_output.force.json`

Smoke result: `aggregate_status=FAIL`, `case_count=1`, `matched_hour_count=1`, `passed_test_count=2`, `failed_test_count=14`. Passing fields: `LU_INDEX`, `TSK`. The suite failure is expected current model behavior, not a schema/runtime failure.

Validation result: `pytest tests/savepoint/ tests/regression/` -> 3 passed, 3 xfailed in 506.42s.

## Unresolved Risks

- `canary_20260521_72h_d02` is blocked because the available L2 CPU WRF directory has only 20 d02 hourly `wrfout` files, not a complete 72h Oracle.
- `idealized_warm_bubble` and `idealized_density_current` are blocked because stock CPU WRF idealized Oracle outputs are not present.
- The milestone wrapper is syntax-checked but not full-run here because `run_milestone_regression.sh M11` intentionally forces full GPU runs for every unblocked case.

## Next Decision Needed

Decide whether to schedule the missing CPU WRF 72h/idealized Oracle generation now, or allow M11/M12/M13 to use the available d02/d03 Canary 24h cases plus blocked-case accounting until those Oracles exist.
