# Worker Report

Summary: Implemented the explicit CUDA Tile M2 bakeoff candidate for the analytic stencil and column fixtures. The kernels compile for sm_120 through the local NVHPC `nvc++ -cuda -gpu=cc120` fallback after `nvcc -arch=sm_120` hits a CUDA 13.1 + glibc/GCC 15 `rsqrt` prototype conflict. Both fixture comparisons pass. `ncu` was invoked for both problems, but this user lacks NVIDIA performance-counter permission (`ERR_NVGPUCTRPERM`), so profile JSONs record that limitation and use `cuobjdump` resource usage plus CUDA occupancy API for available metrics.

## Files Changed

- `src/gpuwrf/backends/cuda_tile/__init__.py`
- `src/gpuwrf/backends/cuda_tile/stencil.cu`
- `src/gpuwrf/backends/cuda_tile/column.cu`
- `src/gpuwrf/backends/cuda_tile/host.cpp`
- `src/gpuwrf/backends/cuda_tile/Makefile`
- `src/gpuwrf/backends/cuda_tile/build.sh`
- `scripts/m2_run_cuda_tile.sh`
- `artifacts/m2/cuda_tile/stencil_profile.json`
- `artifacts/m2/cuda_tile/column_profile.json`
- `artifacts/m2/cuda_tile/correctness.json`
- `artifacts/m2/cuda_tile/maintainability.md`
- `artifacts/m2/cuda_tile/agent_success.json`
- `tests/test_m2_cuda_tile.py`
- `.agent/sprints/2026-05-19-m2-cuda-tile-stencil-column/worker-report.md`

## Commands Run

- `source /home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/env_wrf_gpu.sh`: output `env sourced: CUDA_HOME=unset NVHPC=unset`.
- `bash src/gpuwrf/backends/cuda_tile/build.sh`: latest output `make: Nothing to be done for 'all'.` Earlier first-build output shows `nvcc -arch=sm_120` failing on `rsqrt/rsqrtf` exception-spec conflicts, then successful `nvc++ -cuda -gpu=cc120`.
- `cuobjdump --dump-sass data/scratch/cuda_tile/bench | head -1`: blank first line from cuobjdump. Follow-up `sed -n '1,5p'` shows `arch = sm_120`.
- `bash scripts/m2_run_cuda_tile.sh`: stencil output `wall_time_s=0.000662047982216`, `kernel_launches=1`, `host_device_transfer_bytes=118272`, `theoretical_occupancy_pct=66.666667`; column output `wall_time_s=0.000946304023266`, `kernel_launches=1`, `host_device_transfer_bytes=2560`, `theoretical_occupancy_pct=100`.
- Stencil `python -m gpuwrf.validation.compare_fixture ...`: `pass: true`, all stencil variables pass, `phi_next max_abs_diff=0.0`.
- Column `python -m gpuwrf.validation.compare_fixture ...`: `pass: true`, all column variables pass, largest reported tolerated differences are `qv_next max_abs_diff=4.336808689942018e-19` and `mse_delta max_abs_diff=1.0802470029602773e-12`.
- `python -m json.tool artifacts/m2/cuda_tile/stencil_profile.json`: valid JSON; registers/thread `58`, local memory `0`, occupancy `66.666667`, profiler limitation recorded.
- `python -m json.tool artifacts/m2/cuda_tile/column_profile.json`: valid JSON; registers/thread `24`, local memory `0`, occupancy `100.0`, profiler limitation recorded.
- `python scripts/validate_agentos.py`: `{"ok": true, "errors": [], "required_files_checked": 31, "skills_checked": 13}`.
- `pytest -q`: `65 passed in 8.14s`.
- `python scripts/check_m1_done.py`: `{"ok": true, "errors": [], "sprints_closed": 3}`.
- `python scripts/check_m2_done.py`: exits 1 as expected for partial M2; `candidates_satisfied: 1/6`. The cuda_tile row is present, while other candidates, ADR-001, tester/reviewer reports, and milestone closeout remain outside this worker scope.
- `git ls-files -z | xargs -0 stat -c '%s %n' | sort -nr | head -5`: largest tracked files remain pre-existing PDFs and M1 sample; no new tracked file over 100 KB.
- Deliberate bug check: temporary `data/scratch/cuda_tile/stencil_bug.cu` typo produced `identifier "phi_nxt" is undefined`; source tree was not modified for that check.

## Proof Objects

- Candidate outputs: `data/scratch/cuda_tile/stencil_out.npz`, `data/scratch/cuda_tile/column_out.npz`.
- Parsed profiles: `artifacts/m2/cuda_tile/stencil_profile.json`, `artifacts/m2/cuda_tile/column_profile.json`.
- Correctness aggregate: `artifacts/m2/cuda_tile/correctness.json`.
- Profiler logs: `data/profiler_artifacts/cuda_tile/stencil_ncu_stdout.txt`, `data/profiler_artifacts/cuda_tile/column_ncu_stdout.txt`, matching `*_stderr.txt` and `*_exit.txt`.
- Resource audit: `data/scratch/cuda_tile/resource_usage.txt`.
- Maintainability and agent success: `artifacts/m2/cuda_tile/maintainability.md`, `artifacts/m2/cuda_tile/agent_success.json`.

## Risks

- `ncu` report files were not produced because performance counters are restricted for this user. The script still invokes `ncu` and preserves logs; reviewer/manager must decide whether this is acceptable under the contract risk fallback or whether to rerun with counter permissions.
- The build uses `nvc++ -cuda` fallback after an `nvcc` toolchain/header failure. The produced binary contains sm_120 SASS, but this deviates from the preferred `nvcc` path and is documented in the profile and maintainability artifacts.
- The NPZ reader/writer is intentionally minimal and supports only the dtypes/shapes needed for the M1 analytic fixtures.

## Handoff

Objective complete for the cuda_tile candidate: both bakeoff problems are implemented, correctness passes, profile-quality JSONs exist with honest profiler limitation fields, and tests pass. Next decision needed: reviewer should decide whether the `ERR_NVGPUCTRPERM` fallback evidence is acceptable or whether the manager should rerun this branch with NVIDIA performance-counter permissions enabled before comparing candidates.
