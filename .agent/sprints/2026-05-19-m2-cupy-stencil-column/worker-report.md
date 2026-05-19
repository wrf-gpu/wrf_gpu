# Worker Report

Summary: Implemented the M2 `cupy_or_numba` candidate with CuPy RawKernel kernels for both required fixtures. No Numba fallback was used. The script creates/reuses `data/scratch/m2-cupy-venv/`, installs pinned `cupy-cuda13x==14.0.1`, runs one raw CUDA kernel launch per problem, writes candidate NPZs, and emits M2 profile/correctness artifacts. Correctness passes for both M1 analytic fixtures.

## Files Changed

- `src/gpuwrf/backends/cupy/__init__.py`
- `src/gpuwrf/backends/cupy/stencil.py`
- `src/gpuwrf/backends/cupy/column.py`
- `src/gpuwrf/backends/cupy/bench.py`
- `scripts/m2_run_cupy.sh`
- `artifacts/m2/cupy_or_numba/stencil_profile.json`
- `artifacts/m2/cupy_or_numba/column_profile.json`
- `artifacts/m2/cupy_or_numba/correctness.json`
- `artifacts/m2/cupy_or_numba/maintainability.md`
- `artifacts/m2/cupy_or_numba/agent_success.json`
- `tests/test_m2_cupy.py`
- `.agent/sprints/2026-05-19-m2-cupy-stencil-column/worker-report.md`

## Commands Run

- `bash scripts/m2_run_cupy.sh`
  - stdout: `13000`
  - stderr: none
  - result: exit 0; idempotent rerun reused the venv and rewrote artifacts.
- `python -m gpuwrf.validation.compare_fixture --manifest fixtures/manifests/analytic-stencil-3d-advdiff-v1.yaml --candidate data/scratch/m2-cupy/stencil_out.npz --reference fixtures/samples/analytic-stencil-3d-advdiff-v1.npz`
  - output: `pass: true`, `first_failure: null`; all stencil variables pass.
- `python -m gpuwrf.validation.compare_fixture --manifest fixtures/manifests/analytic-column-thermo-v1.yaml --candidate data/scratch/m2-cupy/column_out.npz --reference fixtures/samples/analytic-column-thermo-v1.npz`
  - output: `pass: true`, `first_failure: null`; all column variables pass.
- `python -m json.tool artifacts/m2/cupy_or_numba/stencil_profile.json`
  - output: valid JSON; `kernel_launches=1`, `registers_per_thread=58`, `local_memory_bytes=64`, `occupancy_pct=66.66666666666667`, `host_device_transfer_bytes=118272`.
- `python -m json.tool artifacts/m2/cupy_or_numba/column_profile.json`
  - output: valid JSON; `kernel_launches=1`, `registers_per_thread=24`, `local_memory_bytes=0`, `occupancy_pct=100.0`, `host_device_transfer_bytes=2560`.
- `pytest -q`
  - output: `88 passed in 14.47s`.
- `python scripts/validate_agentos.py`
  - output: `{"errors": [], "ok": true, "required_files_checked": 31, "skills_checked": 13}`.
- `python scripts/check_m1_done.py`
  - output: `{"errors": [], "manifest_dir": "fixtures/manifests", "ok": true, "sprints_closed": 3}`.
- `python scripts/check_m2_done.py`
  - output: `candidates_satisfied: 2/6`, `ok: false`. The CuPy row is satisfied; remaining errors are expected outside this worker scope: unclosed tester/reviewer/manager reports for this sprint, missing future candidates (`jax`, `triton`, `gt4py`, `kokkos`), missing ADR-001, missing M2 closeout, and missing tester AI provenance log.
- `git ls-files -z | xargs -0 stat -c '%s %n' | sort -nr | head -5`
  - output top tracked file: `1540850 WRF GPU Porting_ Architecture & Verification.pdf`; largest new tracked source/artifact files are under 100 KB.

## Proof Objects

- `artifacts/m2/cupy_or_numba/stencil_profile.json`
- `artifacts/m2/cupy_or_numba/column_profile.json`
- `artifacts/m2/cupy_or_numba/correctness.json`
- `artifacts/m2/cupy_or_numba/maintainability.md`
- `artifacts/m2/cupy_or_numba/agent_success.json`
- Candidate outputs in `data/scratch/m2-cupy/stencil_out.npz` and `data/scratch/m2-cupy/column_out.npz`
- Profiler fallback logs under `data/profiler_artifacts/cupy_or_numba/`

## Risks

- Nsight Compute was invoked but local permissions still trigger `ERR_NVGPUCTRPERM`; JSON includes `profiler_limitation` and uses RawKernel attributes plus CUDA occupancy API fallback.
- Stencil RawKernel reports 64 bytes local memory. The contract only requires zero local memory for the column kernel, which is met.
- `check_m2_done.py` cannot pass before other M2 candidates, tester/reviewer reports, ADR-001, and closeout exist.

## Handoff

Objective complete for the CuPy worker slice. Next decision needed: tester should verify the kernels are real RawKernels, confirm the venv pin is `cupy-cuda13x==14.0.1`, and independently check `column_profile.json` reports `local_memory_bytes=0`.
