# Reviewer Report

## Findings

- major — The sprint diff contains `scripts/dispatch_role.sh`, outside the contract's file ownership list. The contract limits worker edits to the cuda_tile implementation/artifact/test paths and says any other change requires manager approval (`.agent/sprints/2026-05-19-m2-cuda-tile-stencil-column/sprint-contract.md:38`, `.agent/sprints/2026-05-19-m2-cuda-tile-stencil-column/sprint-contract.md:55`). The out-of-scope line is `scripts/dispatch_role.sh:185`, and it is absent from the worker's files-changed list (`.agent/sprints/2026-05-19-m2-cuda-tile-stencil-column/worker-report.md:7`). This must be separated or explicitly approved before merge.
- major — Profiler evidence does not satisfy AC #9/#10 literally. The contract requires each profile JSON to reference the `*.ncu-rep` report and requires JSON numbers parsed from that report (`.agent/sprints/2026-05-19-m2-cuda-tile-stencil-column/sprint-contract.md:105`, `.agent/sprints/2026-05-19-m2-cuda-tile-stencil-column/sprint-contract.md:106`). Current profile artifacts reference only stdout/stderr/exit/resource logs and carry `ERR_NVGPUCTRPERM` fallback text (`artifacts/m2/cuda_tile/stencil_profile.json:3`, `artifacts/m2/cuda_tile/stencil_profile.json:17`, `artifacts/m2/cuda_tile/column_profile.json:3`, `artifacts/m2/cuda_tile/column_profile.json:17`). The risk section allows documented fallback (`sprint-contract.md:160`), but ADR-001 should not treat these as full ncu counter profiles.
- minor — `achieved_bandwidth_gbps` is computed from host/device transfer bytes divided by kernel event time, not from profiler device-memory throughput (`scripts/m2_run_cuda_tile.sh:112`, `scripts/m2_run_cuda_tile.sh:114`, `scripts/m2_run_cuda_tile.sh:126`). This is acceptable only as a best-effort fallback metric and should not be compared as real achieved bandwidth in the bakeoff.
- note — The column implementation launches one block for the single 40-level fixture (`src/gpuwrf/backends/cuda_tile/column.cu:62`, `src/gpuwrf/backends/cuda_tile/column.cu:63`). That matches the one-column M1 fixture but is not a scalable multi-column layout; keep it out of any broader architecture inference.

## Contract Compliance

Build/correctness/profile-shape/test hygiene are mostly satisfied. I independently confirmed `build.sh` exits 0, the binary contains `arch = sm_120` / `.target sm_120`, both `compare_fixture` commands return `pass: true`, both JSON files parse, `validate_agentos.py` is ok, `check_m1_done.py` is ok, and `pytest -q -k 'not test_cuda_tile_pipeline_artifacts_are_valid'` reports 85 passed / 1 deselected. I avoided re-running the full pipeline because it rewrites tracked profile JSONs.

Accepted fallback compliance: `nvc++ -cuda -gpu=cc120` replaces the preferred `nvcc -arch=sm_120` path after the documented CUDA 13.1/GCC rsqrt conflict (`src/gpuwrf/backends/cuda_tile/build.sh:15`, `src/gpuwrf/backends/cuda_tile/build.sh:18`). The produced binary is sm_120-only, so this is not a correctness blocker.

Not fully compliant until fixed or waived: out-of-scope `scripts/dispatch_role.sh` change, and absent `.ncu-rep` profiler reports.

## Correctness Risks

Correctness risk is low for the two M1 analytic fixtures. The stencil and column candidate outputs pass the oracle comparisons; stencil `phi_next` is exact to the fixture tolerance, and the column residuals are roundoff-scale. The hand-rolled NPZ reader/writer is fixture-specific, but malformed input handling was tested by the tester and is sufficient for this sprint's bounded scope.

## Performance Risks

Performance evidence is useful as a smoke baseline, not as final bakeoff-grade counter evidence. Registers/local memory come from `cuobjdump`, and kernel launch count/transfer bytes/wall time come from the bench JSON, but no ncu counter report exists. The reported bandwidth is derived, not measured. The column kernel's single-block fixture mapping also means these numbers should not be extrapolated to realistic column batches.

## Required Fixes

- Before merge, remove `scripts/dispatch_role.sh` from this sprint diff or attach explicit manager approval that this infrastructure change belongs in the sprint branch.
- Before ADR-001 uses cuda_tile performance numbers, rerun this candidate with NVIDIA performance-counter permission so `*.ncu-rep` files exist and profile metrics are parsed from them, or record an explicit manager/reviewer waiver that cuda_tile enters the candidate matrix with documented profiler limitation rather than full ncu evidence.
- In ADR-001 candidate comparison, label `achieved_bandwidth_gbps` for this run as fallback-derived, not profiler-measured.

## Decision

Decision: Accept with required fixes
