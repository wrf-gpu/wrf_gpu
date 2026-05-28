# Worker Report — M14 Lateral Boundary Completeness

## Verdict

`M14_PARTIAL` — boundary-strip specified-zone pass count: **11/11**.

## Objective

Extend lateral-boundary application beyond U/V/T/QVAPOR/PH/MU to cover W, P, PB, PHB, and MUB; verify five-cell WRF width parity; emit boundary-strip and interior/boundary split proofs.

## Files Changed

- `src/gpuwrf/coupling/boundary_apply.py`
- `src/gpuwrf/contracts/state.py`
- `src/gpuwrf/contracts/precision.py`
- `src/gpuwrf/integration/d02_replay.py`
- `src/gpuwrf/coupling/driver.py`
- `src/gpuwrf/io/boundary_replay.py`
- `src/gpuwrf/diagnostics/comprehensive_harness.py`
- `tests/test_m6_boundary_apply.py`
- `tests/test_m6_state_extension.py`
- `.agent/sprints/2026-05-28-m14-lateral-bc-completeness/generate_m14_proofs.py`
- `proofs/m14/**`

Note: `src/gpuwrf/runtime/operational_mode.py` is dirty from another worker and was not staged for this report.

## Commands Run

- `python -m py_compile src/gpuwrf/contracts/state.py src/gpuwrf/contracts/precision.py src/gpuwrf/coupling/boundary_apply.py src/gpuwrf/integration/d02_replay.py src/gpuwrf/coupling/driver.py src/gpuwrf/io/boundary_replay.py`
- `taskset -c 0-3 pytest -q tests/test_m6_boundary_apply.py tests/test_m6_state_extension.py tests/test_m6_precision_matrix.py`
- `taskset -c 0-3 python scripts/m7_daily_pipeline.py --hours 1 --output-dir /tmp/m14_lateral_bc_20260521 --proof-dir proofs/m14`
- `taskset -c 0-3 python .agent/sprints/2026-05-28-m14-lateral-bc-completeness/generate_m14_proofs.py`
- `taskset -c 0-3 pytest -q tests/savepoint/test_dycore_100_steps.py`
- `taskset -c 0-3 python scripts/run_diagnostic_harness.py --hours 1 --radiation-cadence-steps 999999 --output proofs/m14/diagnostic_harness_1h_no_radiation.json`

## Proof Objects Produced

- `proofs/m14/boundary_strip_parity.json`: **11/11 PASS** for WRF specified-zone boundary row vs d02 hourly side-history replay.
- `proofs/m14/relax_zone_width_parity.json`: **PASS** for `spec_bdy_width=5` across all M14 boundary leaves; `wrfbdy_d02` is missing, available `wrfbdy_d01` width is 5.
- `proofs/m14/interior_vs_boundary_split.json`: **M14_BLOCKED** because no M14 hour-1 wrfout was produced.
- `proofs/m14/pipeline_run_20260521.json`: **PIPELINE_BLOCKED**, nonfinite model state after forecast hour 1.
- `proofs/m14/dycore_100_steps_pytest.json`: **PASS**, 1 savepoint test passed in 468.54 s.
- `proofs/m14/diagnostic_harness_1h_no_radiation.json`: **DIAGNOSIS_PRODUCED**, `lateral_boundary` verdict **ACTIVE**.

## Acceptance Notes

- AC1 complete: `apply_lateral_boundaries` now applies U, V, W, theta/T, QVAPOR, P, PB, PH, PHB, MU, and MUB through total/perturbation/base split fields.
- AC2 complete against available evidence: namelist and `wrfbdy_d01` both show five-cell width; all M14 boundary leaves have width 5.
- AC3 partial: `wrfbdy_d02` is not present under the pinned Canary run tree, so the proof uses d02 hourly side-history replay as the oracle and documents the missing `wrfbdy_d02`.
- AC4 blocked: the 1h pipeline failed before writing hour-1 output, so boundary-vs-interior RMSE improvement could not be evaluated.
- AC5 complete: 100-step savepoint parity passed.
- AC6 complete: diagnostic harness re-ran with radiation cadence disabled and `lateral_boundary` remained `ACTIVE`.

## Unresolved Risks

- The forecast still becomes nonfinite by hour 1; M14 did not prove an hour-1 boundary/interior RMSE improvement.
- `wrfbdy_d02` is absent for the pinned run, so decoded native wrfbdy proof for P/PB/PHB/MUB is unavailable.
- Boundary base fields PB/PHB/MUB are packed as one time record because they are invariant; this avoids carrying redundant 25-hour base strips through the scan.

## Next Decision Needed

Decide whether the next sprint should investigate the hour-1 nonfinite pressure/mass blow-up now exposed by the complete boundary application, or first add a native `wrfbdy_d02`/nest-boundary fixture if that file can be generated.
