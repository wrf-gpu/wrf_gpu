# Worker Report — M6.x ADR-023 Production Grade

Summary: Implemented the production-grade ADR-023 follow-up within the worker scope. MPAS slice trajectory RMSE improved from the prototype's 0.387407 to 0.016867 at default `epssm=0.1`, below the 0.15 gate. R7 remains GREEN, warm bubble remains `PASS_WARM_BUBBLE_600S`, nonhydrostatic `mu_continuity` now executes inside the scan body, and F2/F5/F7/F9 are folded into ADR/code/tests. ADR status remains PROPOSED pending reviewer concurrence.

## Files Changed

- `.agent/decisions/ADR-023-conservative-column-solver.md`
- `src/gpuwrf/dynamics/acoustic_wrf.py`
- `src/gpuwrf/dynamics/vertical_implicit_solver.py`
- `tests/test_m6x_adr023_production_grade.py`
- `.agent/sprints/2026-05-23-m6x-adr023-production-grade/worker-report.md`
- Proof text/json files in this sprint folder.

## Commands Run

- `pytest tests/test_m6x_adr023_production_grade.py -v | tee .../proof_production_gate.txt`
  Output: `4 passed in 6.70s`.
- `pytest tests/test_m6x_mpas_column_slice_oracle.py tests/test_m6x_vertical_acoustic_oracle.py tests/test_m6x_adr023_column_solver.py tests/test_m6x_c2_acoustic.py -v | tee .../proof_no_regression.txt`
  Output: `19 passed in 16.02s`.
- `python scripts/m6_warm_bubble_test.py --output .../proof_warm_bubble_production.json | tee .../proof_warm_bubble_production.txt`
  Output: `PASS_WARM_BUBBLE_600S`, 600 s `w_max=8.523914985976297`, centroid `3385.2273071323343`, no nonfinite step.
- `pytest tests/test_m3_transfer_audit.py tests/test_m6x_c2_acoustic.py::test_acoustic_scan_jaxpr_has_scan_and_no_host_callbacks -v | tee .../proof_transfer_audit.txt`
  Output: `5 passed in 2.62s`.
- epssm sweep proof script: `epssm=0.0` R7 pass / slice RMSE `0.019062`; `epssm=0.1` R7 pass / slice RMSE `0.016867`; `epssm=0.3` R7 fail / slice RMSE `0.013125`. Default remains `0.1`.
- `nsys profile --capture-range=cudaProfilerApi ...` warmed vertical operator probe, summarized in `proof_launch_count_production.txt`.
  Output summary: `cuGraphLaunch_calls=2`, `cuLaunchKernelEx_calls=67`, kernel rows `loop_add_fusion:32`, `loop_divide_fusion:16`, `loop_subtract_fusion:16`, reverse/slice 3 total.

## Proof Objects

- `.agent/sprints/2026-05-23-m6x-adr023-production-grade/proof_production_gate.txt`
- `.agent/sprints/2026-05-23-m6x-adr023-production-grade/proof_no_regression.txt`
- `.agent/sprints/2026-05-23-m6x-adr023-production-grade/proof_warm_bubble_production.txt`
- `.agent/sprints/2026-05-23-m6x-adr023-production-grade/proof_warm_bubble_production.json`
- `.agent/sprints/2026-05-23-m6x-adr023-production-grade/proof_transfer_audit.txt`
- `.agent/sprints/2026-05-23-m6x-adr023-production-grade/proof_epssm_sweep.txt`
- `.agent/sprints/2026-05-23-m6x-adr023-production-grade/proof_launch_count_production.txt`

## Risks

- Launch count exceeded the prototype target: warmed profiler capture reports 67 CUDA kernel instances for the MPAS recurrence path versus prototype baseline 20. Correctness gates pass, but optimization is still required before performance claims.
- The coupled warm-bubble path still uses the documented positive-updraft damping analogue and a small positive-mass CFL limiter on in-scan `mu_continuity`. This is honest production-grade stabilization for the current proof ladder, not a 24h forecast physics claim.
- The MPAS slice remains a symbolic source-derived oracle, not a built MPAS executable savepoint.

## Handoff

Objective: promote ADR-023 from prototype to production-grade evidence for the current ladder rung without expanding `AcousticScanCarry` or adding Newton.

Files changed: listed above.

Commands run: listed above with outputs.

Proof objects produced: listed above.

Unresolved risks: launch-count regression, warm-bubble limiter/damping relevance, and MPAS symbolic-slice limitation.

Next decision needed: reviewer should decide whether the residual limiter/launch risks are acceptable for ADR-023 to proceed, or whether the next sprint must be an optimization/fallback amendment before d02 boundary replay.
