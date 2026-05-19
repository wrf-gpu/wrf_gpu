# Worker Report — M2 Triton Stencil/Column

Summary: Implemented the attempt-2 fix required by the amended contract. The Triton bench now derives `registers_per_thread` and `local_memory_bytes` from `cuobjdump --dump-resource-usage` sections matching each problem's own kernel symbol (`_stencil_advdiff_kernel` or `_column_thermo_kernel`) instead of taking maxima across all recently cached cubins. Regenerated Triton artifacts report stencil REG:60 LOCAL:0 and column REG:34 LOCAL:0, with the column profile no longer contaminated by the stencil cubin.

## Objective

Repair the M2 Triton candidate so profile JSON resource metrics are kernel-symbol-aware and the binding tester edge case `test_column_profile_registers_match_column_kernel_cubin` passes, without expanding beyond the sprint contract.

## Files changed

- `src/gpuwrf/backends/triton/bench.py`
- `artifacts/m2/triton/stencil_profile.json`
- `artifacts/m2/triton/column_profile.json`
- `artifacts/m2/triton/correctness.json`
- `artifacts/m2/triton/agent_success.json`
- `.agent/sprints/2026-05-19-m2-triton-stencil-column/worker-report.md`

## Commands run and output

- `bash scripts/m2_run_triton.sh`
  - stdout: `3.7.0 13.0`
  - stderr: empty
  - exit: 0

- `python -m gpuwrf.validation.compare_fixture --manifest fixtures/manifests/analytic-stencil-3d-advdiff-v1.yaml --candidate data/scratch/m2-triton/stencil_out.npz --reference fixtures/samples/analytic-stencil-3d-advdiff-v1.npz`
  - stdout: JSON with `"fixture_id": "analytic-stencil-3d-advdiff-v1"`, `"pass": true`, all variables passing with `max_abs_diff: 0.0`.
  - stderr: empty
  - exit: 0

- `python -m gpuwrf.validation.compare_fixture --manifest fixtures/manifests/analytic-column-thermo-v1.yaml --candidate data/scratch/m2-triton/column_out.npz --reference fixtures/samples/analytic-column-thermo-v1.npz`
  - stdout: JSON with `"fixture_id": "analytic-column-thermo-v1"`, `"pass": true`; largest reported differences were `qv_next max_abs_diff=4.336808689942018e-19` and `mse_delta max_abs_diff=1.0802470029602773e-12`, both within tolerance.
  - stderr: empty
  - exit: 0

- `python -m json.tool artifacts/m2/triton/stencil_profile.json`
  - stdout: valid JSON; key values `registers_per_thread=60`, `local_memory_bytes=0`, `occupancy_pct=66.66666666666667`, `kernel_launches=1`, `wall_time_s=2.525e-05`, `achieved_bandwidth_method="fallback-derived"`.
  - stderr: empty
  - exit: 0

- `python -m json.tool artifacts/m2/triton/column_profile.json`
  - stdout: valid JSON; key values `registers_per_thread=34`, `local_memory_bytes=0`, `occupancy_pct=100.0`, `kernel_launches=1`, `wall_time_s=2.7e-05`, `achieved_bandwidth_method="fallback-derived"`.
  - stderr: empty
  - exit: 0

- `python -m pytest -q tests/test_m2_triton_edge_cases.py::test_column_profile_registers_match_column_kernel_cubin tests/test_m2_triton_edge_cases.py::test_stencil_profile_registers_match_stencil_kernel_cubin tests/test_m2_triton_edge_cases.py::test_cuobjdump_resource_usage_artifact_has_kernel_section`
  - stdout: `3 passed in 0.02s`
  - stderr: empty
  - exit: 0

- `python -m pytest -q tests/test_m2_triton_edge_cases.py::test_column_profile_registers_match_column_kernel_cubin tests/test_m2_triton_edge_cases.py::test_stencil_profile_registers_match_stencil_kernel_cubin tests/test_m2_triton_edge_cases.py::test_cuobjdump_resource_usage_artifact_has_kernel_section tests/test_m2_triton_edge_cases.py::test_agent_success_log_is_well_formed`
  - stdout: `4 passed in 0.02s`
  - stderr: empty
  - exit: 0

- `pytest -q`
  - stdout: `233 passed in 119.55s (0:01:59)`
  - stderr: empty
  - exit: 0

- `pytest -q tests/test_m2_triton_edge_cases.py::test_agent_success_log_is_well_formed`
  - stdout: `1 passed in 0.02s`
  - stderr: empty
  - exit: 0

- `python scripts/validate_agentos.py`
  - stdout: `{"errors": [], "ok": true, "required_files_checked": 31, "skills_checked": 13}`
  - stderr: empty
  - exit: 0

- `python scripts/check_m1_done.py`
  - stdout: `{"errors": [], "manifest_dir": "fixtures/manifests", "ok": true, "sprints_closed": 3}`
  - stderr: empty
  - exit: 0

- `python scripts/check_m2_done.py`
  - stdout: `{"candidates_satisfied": 5, "candidates_total": 6, "ok": false, ...}` with errors for this sprint not yet closed before this report existed, missing `artifacts/m2/gt4py/`, missing ADR-001, missing M2 closeout, and `M2-backend-bakeoff.md` reviewer decision not accepted.
  - stderr: empty
  - exit: 1

- `git ls-files -z | xargs -0 stat -c '%s %n' | sort -nr | head -5`
  - stdout top five: `1540850 WRF GPU Porting_ Architecture & Verification.pdf`; `97515 wrf to gpu gpt5.5 deep research.pdf`; `61080 fixtures/samples/analytic-stencil-3d-advdiff-v1.npz`; `38916 tests/test_m2_triton_edge_cases.py`; `31304 tests/test_m2_jax_edge_cases.py`.
  - stderr: empty
  - exit: 0

## Proof objects produced

- `artifacts/m2/triton/stencil_profile.json`
- `artifacts/m2/triton/column_profile.json`
- `artifacts/m2/triton/correctness.json`
- `artifacts/m2/triton/agent_success.json`
- `data/profiler_artifacts/triton/stencil_cuobjdump_resource_usage.txt`
- `data/profiler_artifacts/triton/column_cuobjdump_resource_usage.txt`
- `data/profiler_artifacts/triton/stencil_triton_0.cubin`
- `data/profiler_artifacts/triton/column_triton_0.cubin`
- `data/scratch/m2-triton/stencil_out.npz`
- `data/scratch/m2-triton/column_out.npz`

## Unresolved risks

- Nsight Compute profiler artifacts are still limited by local performance-counter permissions; the sprint uses the approved cuobjdump/fallback-derived resource and bandwidth path.
- `python scripts/check_m2_done.py` cannot be green from this worker alone because M2-wide GT4Py, ADR-001, M2 closeout, and reviewer-acceptance artifacts remain manager/reviewer scope. Before this report existed it also flagged `missing worker-report.md`; this report resolves only that worker-owned part.

## Handoff

Next decision needed: tester/reviewer should re-run the edge-case cubin checks and review the attempt-2 diff. Manager should handle M2-wide closeout blockers outside this worker's file ownership.
