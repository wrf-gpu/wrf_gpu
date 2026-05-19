# Worker Report

## Summary

Summary: Implemented the M2 Kokkos backend candidate for both analytic bakeoff problems using Kokkos 4.7.1 `Kokkos::View` state and CUDA execution on RTX 5090 / compute capability 12.0. The bench builds with the required local Kokkos source/install cache under `data/scratch/kokkos-{src,install}`, emits a usage message with no args, and produces `sm_120` SASS. Both fixture comparisons pass. Nsight Compute was invoked for both problems but exits with the known local `ERR_NVGPUCTRPERM`; profile JSONs use the same fallback pattern as earlier M2 candidates: registers/local memory from `cuobjdump`, runtime/device facts from the bench, and wall time/transfer bytes from bench output.

## Files Changed

- `src/gpuwrf/backends/kokkos/__init__.py`
- `src/gpuwrf/backends/kokkos/stencil.cpp`
- `src/gpuwrf/backends/kokkos/column.cpp`
- `src/gpuwrf/backends/kokkos/host.cpp`
- `src/gpuwrf/backends/kokkos/CMakeLists.txt`
- `src/gpuwrf/backends/kokkos/build.sh`
- `scripts/m2_run_kokkos.sh`
- `tests/test_m2_kokkos.py`
- `artifacts/m2/kokkos/stencil_profile.json`
- `artifacts/m2/kokkos/column_profile.json`
- `artifacts/m2/kokkos/correctness.json`
- `artifacts/m2/kokkos/maintainability.md`
- `artifacts/m2/kokkos/agent_success.json`
- `.agent/sprints/2026-05-19-m2-kokkos-stencil-column/worker-report.md`

## Commands Run

- `source /home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/env_wrf_gpu.sh`: no output; environment loaded.
- `bash src/gpuwrf/backends/kokkos/build.sh`: first run cloned Kokkos tag `4.7.01`, configured `Kokkos_ENABLE_CUDA=ON`, `Kokkos_ARCH_BLACKWELL120=ON`, `Kokkos_ENABLE_CUDA_LAMBDA=ON`, installed to `data/scratch/kokkos-install/`, built `data/scratch/kokkos/bench`; cached validation rerun output ended with `[100%] Built target bench`.
- `data/scratch/kokkos/bench`: exit 0, output `usage: bench stencil|column --input path --output path` and `bench config`.
- `cuobjdump --dump-sass data/scratch/kokkos/bench | grep -m1 'arch'`: `arch = sm_120`.
- `bash scripts/m2_run_kokkos.sh`: exit 0. Stencil run JSON: `wall_time_s=9.342e-05`, `kernel_launches=1`, `host_device_transfer_bytes=118272`, `kokkos_execution_space=Cuda`, `runtime_compute_capability=12.0`. Column run JSON: `wall_time_s=0.000155621`, `kernel_launches=1`, `host_device_transfer_bytes=2560`, same runtime facts.
- `python -m gpuwrf.validation.compare_fixture --manifest fixtures/manifests/analytic-stencil-3d-advdiff-v1.yaml --candidate data/scratch/kokkos/stencil_out.npz --reference fixtures/samples/analytic-stencil-3d-advdiff-v1.npz`: `pass: true`.
- `python -m gpuwrf.validation.compare_fixture --manifest fixtures/manifests/analytic-column-thermo-v1.yaml --candidate data/scratch/kokkos/column_out.npz --reference fixtures/samples/analytic-column-thermo-v1.npz`: `pass: true`.
- `python -m json.tool artifacts/m2/kokkos/stencil_profile.json`: valid JSON; registers 64, local memory 0, occupancy 66.6667, one launch.
- `python -m json.tool artifacts/m2/kokkos/column_profile.json`: valid JSON; registers 40, local memory 0, occupancy 100.0, one launch.
- `pytest -q`: `114 passed in 29.91s`.
- `python scripts/validate_agentos.py`: `{"errors": [], "ok": true, "required_files_checked": 31, "skills_checked": 13}`.
- `python scripts/check_m1_done.py`: `{"errors": [], "manifest_dir": "fixtures/manifests", "ok": true, "sprints_closed": 3}`.
- `python scripts/check_m2_done.py`: exits 1 as expected before remaining M2 candidates and other roles; reports `candidates_satisfied: 3` of 6 and no Kokkos artifact coverage error. Remaining errors are missing JAX/Triton/GT4Py artifacts, ADR/closeout, and tester/reviewer provenance/report closure owned by later roles.
- `git ls-files -z | xargs -0 stat -c '%s %n' | sort -nr | head -5`: largest tracked files are pre-existing PDFs and fixture sample; no new tracked file exceeds 100 KB.
- `sed 's/phi_next(idx)/phi_nxt(idx)/' ... && data/scratch/kokkos-install/bin/nvcc_wrapper ...`: deliberate scratch bug compile exits 2; stderr shows the known CUDA 13.1/GCC 15 `rsqrt` conflict followed by `identifier "phi_nxt" is undefined`.

## Proof Objects

- `artifacts/m2/kokkos/stencil_profile.json`
- `artifacts/m2/kokkos/column_profile.json`
- `artifacts/m2/kokkos/correctness.json`
- `artifacts/m2/kokkos/maintainability.md`
- `artifacts/m2/kokkos/agent_success.json`
- External/generated logs under `data/scratch/kokkos/` and `data/profiler_artifacts/kokkos/`, including `resource_usage.txt`, `kokkos_config.txt`, `*_run.json`, `*_ncu_*`, and deliberate bug logs.

## Risks

- `ncu` counter access is still blocked by `ERR_NVGPUCTRPERM`; this sprint uses the manager-approved M2 fallback evidence path.
- Stencil register count is exactly at the contract limit (`64`), so small future changes could violate the Kokkos row unless watched.
- The column Kokkos TeamPolicy kernel reports 1032 bytes shared memory from Kokkos runtime scaffolding but `LOCAL:0`; the sprint AC is local memory bytes, which passes.
- `check_m2_done.py` cannot be green from this worker alone because other candidate families, ADR-001, tester, reviewer, and manager closeout are out of scope.

## Handoff

Objective complete for the Kokkos M2 candidate. Files are ready for tester verification of CUDA-space `View` allocation, fresh linkage, and resource parsing. Next decision needed: tester/reviewer should decide whether the fallback profiler evidence and exact-limit stencil register pressure are acceptable for ADR-001 comparison.
