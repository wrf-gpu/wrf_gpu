# Oracle-Baseline Regression Suite

This suite treats CPU WRF output as the Oracle and compares milestone GPU
forecast output field by field, output time by output time. It is intentionally
allowed to fail today. The target is a fixed regression ratchet: every milestone
gets a snapshot, and the next milestone is judged for newly passing tests,
newly failing tests, and metric movement that does not cross a pass/fail
boundary.

## Add A Case

Add an entry to `oracle_cases.yaml` with:

- `case_id`
- `type`: `real` or `idealized`
- `oracle.run_dir`, `oracle.domain`, and expected forecast hours
- `expected_variables`
- `tolerance_class`
- `gpu.runner` metadata for the existing GPU pipeline

Do not commit WRF output. Reference external CPU WRF runs by path and record the
generation command when a required Oracle is missing. Missing Oracles stay in
the manifest as `status: BLOCKED`.

## Add A Tolerance Class

Add a class to `tolerances.yaml`. Tolerances are predeclared pass criteria, not
current-achievement numbers. `EQUIVALENCE_LOOSE` may inherit another class and
scale numeric thresholds, but widening an existing class requires an ADR.

## Run It

Fast smoke:

```bash
taskset -c 0-3 python scripts/run_regression_suite.py --smoke
```

Milestone snapshot:

```bash
taskset -c 0-3 bash scripts/run_milestone_regression.sh M11
```

The milestone wrapper writes `proofs/regression/snapshot_M11.json` and compares
it with the previous snapshot, for example `snapshot_M10.json`, producing
`proofs/regression/regression_check_M10_to_M11.json`.

## Output Excerpt

```json
{
  "schema": "OracleBaselineRegressionAggregate",
  "milestone": "M11",
  "aggregate_status": "FAIL",
  "passed_test_count": 3,
  "failed_test_count": 13,
  "blocked_case_count": 3
}
```

Each per-case JSON under `proofs/regression/` includes per-time field metrics,
per-field tolerance decisions, GPU run provenance, and CPU WRF Oracle paths.
