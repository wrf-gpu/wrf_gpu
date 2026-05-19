# Sprint Contract

Sprint ID: `2026-05-19-m2-kokkos-stencil-column`
Milestone: M2 — Backend Bakeoff
Sequence: S4 (per M2-S1 readiness ranking: kokkos is 3rd — verdict=go-with-version-bump, requires Kokkos 4.7.1 source build with `Kokkos_ARCH_BLACKWELL120=ON`)
Worker: gpt-kernel-worker (Codex `gpt-5.5` `high`)
Tester: sonnet-test-engineer (Claude Opus 4.7 `xhigh` — cross-AI verification)
Reviewer: opus-reviewer (Codex `gpt-5.5` `high`)
Candidate family: `kokkos` (C++ performance-portability library, the SCREAM/E3SM exascale path)
Approval status: opened 2026-05-19 by manager after M2-S3 closeout.

## Objective

Implement both bakeoff problems in **C++ using Kokkos** parallel-for + parallel-reduce abstractions with `Kokkos::View` for state, targeting Kokkos's CUDA backend on RTX 5090 (cc120). Produces the same profile/correctness/maintainability/agent-success artifacts as M2-S2 and M2-S3 for direct cross-candidate comparison.

This is the highest-stakes candidate for the ADR-001 decision because:
1. Kokkos is the path that delivered SCREAM at exascale (1.26 SYPD on Frontier) — the only documented atmospheric NWP rewrite that broke the 5× ceiling at scale.
2. Kokkos is the only candidate that's truly vendor-agnostic (NVIDIA + AMD + Intel + CPU).
3. It is also the most verbose and has the steepest agent-iteration friction — `agent_success.json` for this sprint is a particularly important data point.

Same two problems (definitions in `src/gpuwrf/fixtures/analytic.py`):
- **Problem 1**: 3D advection-diffusion stencil, 32×16×8 grid, fp64.
- **Problem 2**: register-heavy thermo column, 40-level column, fp64.

## Non-Goals

- No JAX, Triton, cuda_tile, cupy, gt4py — other M2 sprints.
- No multi-backend Kokkos exercise (CUDA backend only; AMD/CPU is a future M2.x or M3 concern).
- No `Kokkos::Cuda::SharedSpace` manual tiling beyond what's natural for the stencil. Worker uses idiomatic `MDRangePolicy` + `team_policy` patterns first; only drops to shared-memory tiling if the column kernel spills registers.
- No mixed precision.
- No multi-GPU.

## File Ownership

Worker may create or edit only these paths:

- `src/gpuwrf/backends/kokkos/__init__.py` (new if missing — package marker for tests)
- `src/gpuwrf/backends/kokkos/stencil.cpp` (new — Problem 1)
- `src/gpuwrf/backends/kokkos/column.cpp` (new — Problem 2)
- `src/gpuwrf/backends/kokkos/host.cpp` (new — NPZ reader + driver, like cuda_tile/host.cpp)
- `src/gpuwrf/backends/kokkos/CMakeLists.txt` (new — finds Kokkos via `find_package(Kokkos)`)
- `src/gpuwrf/backends/kokkos/build.sh` (new — clone Kokkos 4.7.1 if missing, build with CUDA + BLACKWELL120, then build the bench)
- `scripts/m2_run_kokkos.sh` (new — full pipeline: source NVHPC env, build Kokkos (idempotent — cached at `data/scratch/kokkos-install/`), build bench, run, parse to JSON)
- `artifacts/m2/kokkos/stencil_profile.json` (new)
- `artifacts/m2/kokkos/column_profile.json` (new)
- `artifacts/m2/kokkos/correctness.json` (new)
- `artifacts/m2/kokkos/maintainability.md` (new, ≤300 words)
- `artifacts/m2/kokkos/agent_success.json` (new)
- `tests/test_m2_kokkos.py` (new)
- `pyproject.toml` (do NOT add C++/CMake deps; everything lives in `data/scratch/`)

Any change outside this list requires manager approval.

## Inputs

- M1 fixtures on main (analytic-stencil + analytic-column).
- M1 comparison CLI: `src/gpuwrf/validation/compare_fixture.py`.
- M2-S1 scout pin: Kokkos 4.7.1 with `-DKokkos_ARCH_BLACKWELL120=ON`.
- M2-S2 cuda_tile (`src/gpuwrf/backends/cuda_tile/`) — useful comparison reference for kernel structure + NPZ I/O patterns.
- Project memory `project_target_hardware.md` — note the ncu permission limitation + the nvcc/nvc++ -cuda workaround for CUDA 13.1 + GCC 15 headers (kokkos may hit the same issue via its CUDA backend).
- NVHPC env: `source /home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/env_wrf_gpu.sh`.

## Acceptance Criteria

All must hold.

### Build & install
1. `bash src/gpuwrf/backends/kokkos/build.sh` is idempotent: first run clones Kokkos 4.7.1 to `data/scratch/kokkos-src/`, builds with `-DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_BLACKWELL120=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON` and installs to `data/scratch/kokkos-install/`, then builds the bench. Subsequent runs see the install and skip rebuilds.
2. Bench executable at `data/scratch/kokkos/bench` exists and exits cleanly when run with no args (prints usage).
3. SASS check: `cuobjdump --dump-sass data/scratch/kokkos/bench | grep -m1 'arch'` shows `sm_120` (Blackwell). If only PTX is embedded (CUDA backend may JIT), document this in maintainability.md and verify runtime compute capability via `cudaGetDeviceProperties`.

### Correctness
4. Stencil round-trip: `compare_fixture --pass=true` against M1 stencil fixture.
5. Column round-trip: same.

### Profile JSON (per problem, same schema as cuda_tile + cupy)
6. Both `stencil_profile.json` and `column_profile.json` validate against the `PERFORMANCE_TARGETS.md` schema. Required fields with the `profiler_limitation` + `achieved_bandwidth_method` conventions established by M2-S2.
7. `kernel_launches` per problem ≤ 5 (the whole reason for Kokkos's `parallel_for` is to fuse).
8. **`local_memory_bytes` for column kernel = 0.** (Same critical AC as cuda_tile and cupy.)
9. `registers_per_thread` ≤ 64 stencil, ≤ 128 column.
10. Numbers derived from `cuobjdump` + Kokkos's `kokkos_print_configuration()` runtime info + bench wall-time. Same `ERR_NVGPUCTRPERM` fallback as cuda_tile.

### Maintainability narrative (≤300 words)
11. Covers: build complexity (lines of CMake, time to clone+build Kokkos, disk used at `data/scratch/kokkos-*`), error legibility on a deliberate bug (a 50-page template error counts as "low" legibility), debugger story, agent-iteration friction.

### Agent-success
12. `agent_success.json` populated. This sprint's `sprint_count`/`reviewer_rejections` are particularly informative for ADR-001 — Kokkos is expected to be harder for agents than CuPy or even cuda_tile.

### Tests
13. `tests/test_m2_kokkos.py`: schema validation of both profile JSONs, correctness JSON pass-check, bench-executable presence, SASS contains `sm_120` (or runtime CC check).
14. `pytest -q` passes overall.

### Hygiene
15. `validate_agentos.py` ok.
16. `check_m1_done.py` ok.
17. No file >100 KB committed beyond pre-existing.
18. Kokkos source/install caches at `data/scratch/kokkos-{src,install}/` (gitignored under `data/`).

## Validation Commands

```bash
source /home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/env_wrf_gpu.sh
bash src/gpuwrf/backends/kokkos/build.sh                # idempotent: kokkos + bench
cuobjdump --dump-sass data/scratch/kokkos/bench | grep -m1 'arch'
bash scripts/m2_run_kokkos.sh                           # runs bench, writes JSONs
python -m gpuwrf.validation.compare_fixture --manifest fixtures/manifests/analytic-stencil-3d-advdiff-v1.yaml --candidate data/scratch/kokkos/stencil_out.npz --reference fixtures/samples/analytic-stencil-3d-advdiff-v1.npz
python -m gpuwrf.validation.compare_fixture --manifest fixtures/manifests/analytic-column-thermo-v1.yaml --candidate data/scratch/kokkos/column_out.npz --reference fixtures/samples/analytic-column-thermo-v1.npz
python -m json.tool artifacts/m2/kokkos/stencil_profile.json
python -m json.tool artifacts/m2/kokkos/column_profile.json
pytest -q
python scripts/check_m1_done.py
python scripts/check_m2_done.py    # kokkos row should show satisfied
git ls-files -z | xargs -0 stat -c '%s %n' | sort -nr | head -5
```

## Performance Metrics

Captured in profile JSONs. Same sanity bounds as M2-S2/S3: ≤5 kernel launches, occupancy ≥25% stencil / ≥20% column, registers ≤64/128, local_memory_bytes=0 on column.

ADR-001 will compare across candidates; this sprint just produces the kokkos row.

## Proof Object

- Diff (File Ownership only).
- 5 artifacts in `artifacts/m2/kokkos/`.
- Lifecycle reports.

## Risks

- **Kokkos source build is the longest of any M2 candidate.** Estimate 5–15 min on this workstation. Worker monitors and writes a BLOCKER if it exceeds 30 min.
- **CMake + Kokkos discovery can be brittle.** Worker uses `find_package(Kokkos PATHS data/scratch/kokkos-install)` explicitly rather than relying on system search.
- **CUDA 13.1 + GCC 15 header bug** may bite Kokkos's `nvcc_wrapper`. Workaround: pass `-ccbin nvc++` to nvcc, or use the `nvc++ -cuda` pattern from M2-S2 if needed.
- **Disk usage**: Kokkos source ~200 MB, install ~50 MB, bench small. Worker monitors `df -h /mnt/data`; BLOCKER if below 50 GB.
- **Template error pages** may be 50+ pages on Kokkos compile failures. Worker captures them honestly in maintainability.md for ADR-001's "agent-iteration friction" comparison.

## Handoff Requirements

- Worker pushes to branch `worker/gpt/m2-kokkos-stencil-column`.
- Tester is Claude Opus 4.7 xhigh: verifies the Kokkos build actually finished + linked correctly (not a stale binary), verifies `View` allocations are CUDA-space (not Host), and counts agent-iteration cycles from worker's build logs.
- After reviewer Accept, manager merges to main, pushes, opens M2-S5 (jax).

## Note on manager-during-worker hygiene

Per memory `feedback_manager_autonomy.md` "Operational addition 2026-05-19": manager will NOT commit unrelated files while this worker is in flight to avoid worker-branch contamination.
