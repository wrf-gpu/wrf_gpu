# M10 Worker Report

Verdict: `M10_COMPLETE`

Headline: LU_INDEX is now an int32 `State` leaf populated from `wrfinput_d02`, written through operational output, and bitwise matches the Canary 20260521 WRF reference.

## Objective

Add the missing LU_INDEX static field to the GPU `State`, populate it from the same wrfinput land-state path as XLAND/roughness, verify static-field parity, rerun Canary 20260521 operational output, and confirm no T2/U10/V10 RMSE regression versus post-iter2.

## Files Changed

- `src/gpuwrf/contracts/state.py`
- `src/gpuwrf/contracts/precision.py`
- `src/gpuwrf/integration/d02_replay.py`
- `src/gpuwrf/runtime/operational_mode.py`
- `tests/savepoint/test_static_fields.py`
- `tests/test_m6_precision_matrix.py`
- `tests/test_m6_state_extension.py`
- `proofs/m10/**`
- `.agent/sprints/2026-05-28-m10-lu-index-static-fix/worker-report.md`

## Commands Run

- `taskset -c 0-3 pytest -q tests/savepoint/test_static_fields.py` -> `1 passed`
- `taskset -c 0-3 env PYTHONPATH=src python - <<'PY' ... build_replay_case LU_INDEX check ... PY` -> int32, shape `(66, 159)`, max_abs_diff `0`
- `taskset -c 0-3 pytest -q tests/test_m6_state_extension.py tests/test_m6_precision_matrix.py` -> `4 passed`
- `taskset -c 0-3 pytest -q tests/savepoint/` -> `2 passed, 3 xfailed in 428.27s`
- `taskset -c 0-3 env PYTHONPATH=src JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 python scripts/m7_daily_pipeline.py --run-id 20260521_18z_l3_24h_20260522T133443Z --hours 24 --output-dir /tmp/m10_lu_index_static_fix_20260521 --proof-dir proofs/m10 --run-root /mnt/data/canairy_meteo/runs/wrf_l3 --domain d02` -> 24 wrfout files, inventory PASS, speedup PASS, payload `PIPELINE_PARTIAL` because station score was not requested
- `taskset -c 0-3 env PYTHONPATH=src python scripts/m7_gpu_vs_cpu_skill_diff.py --gpu-root /tmp/m10_lu_index_static_fix_20260521 --cpu-run /mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z --output proofs/m10/post_m10_skill_diff.json --variables T2 U10 V10` -> script status `FAIL_SKILL_DIFF`, M10 non-regression annotation PASS

Intermediate checks also exposed and resolved a missing precision-registry entry for the new state leaf. A mixed pytest command that combined GPU `State.zeros` tests with `tests/savepoint/` failed because the savepoint conftest forces JAX CPU; rerunning those suites separately passed.

## Proof Objects Produced

- `proofs/m10/static_field_parity_after_fix.json`
  - `lu_index`: 24 hourly GPU wrfouts vs CPU WRF hourly wrfouts: max_abs_diff `0`, RMSE `0`, `BITWISE_MATCH`
  - `lu_index_replay_state_vs_wrfinput`: int32 state leaf vs wrfinput: max_abs_diff `0`, RMSE `0`, `BITWISE_MATCH`
  - `hgt`, `landmask`, `xland`, `ivgtyp`, `isltyp`: parity PASS under recorded reference policy
  - `roughness_m`: `DERIVED_MATCH` because ZNT/ROUGHNESS_M is absent from wrfinput
- `proofs/m10/post_m10_skill_diff.json`
  - non-regression vs `.agent/sprints/2026-05-27-m7-skill-fix-iter2/post_iter2_skill_diff.json`: T2/U10/V10 RMSE deltas all `0.0`
- `proofs/m10/pipeline_run_20260521.json`
  - 24-hour operational run completed; 24 wrfout files emitted; finite-state check PASS; wrfout inventory PASS; speedup PASS

## Unresolved Risks

- Pipeline payload remains `PIPELINE_PARTIAL` because station scoring was not requested in the pipeline command; the separate M10 skill diff was run and passed non-regression.
- GPU-vs-CPU skill remains poor in absolute terms (`FAIL_SKILL_DIFF`), unchanged from post-iter2; this sprint only gates non-regression.
- Existing HGT source behavior is not changed: operational writer matches CPU WRF hourly HGT, while wrfinput HGT differs. Recorded in `hgt_wrfinput_delta_existing`; outside LU_INDEX scope.
- `roughness_m` has no direct ZNT/ROUGHNESS_M wrfinput source in this fixture, so parity is against the existing prescribed wrfinput-derived formula.

## Next Decision Needed

No M10 decision needed. A separate sprint should decide whether terrain/HGT should be sourced from wrfinput or remain aligned to CPU WRF hourly output.
