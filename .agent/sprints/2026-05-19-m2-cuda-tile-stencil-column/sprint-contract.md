# Sprint Contract

Sprint ID: `2026-05-19-m2-cuda-tile-stencil-column`
Milestone: M2 — Backend Bakeoff
Sequence: S2 (per M2 runbook readiness order: cuda_tile first — verdict=go, no version-pin gymnastics)
Worker: gpt-kernel-worker (Codex `gpt-5.5` `high`)
Tester: sonnet-test-engineer (Claude Opus 4.7 `xhigh` — cross-AI verification)
Reviewer: opus-reviewer (Codex `gpt-5.5` `high`)
Candidate family: `cuda_tile` (explicit CUDA C++ with shared-memory tile-resident kernels)
Approval status: opened 2026-05-19 by manager after M2-S1 closeout.

## Objective

Implement both bakeoff problems in **explicit CUDA C++** using tile-resident shared-memory kernels, run them on the project's RTX 5090 against M1 fixtures, and emit profiler-quality evidence in the M2-DONE schema. This is the candidate that gives the manager the most direct read on the upper bound of hand-tuned GPU performance — every other candidate's profile will be compared against it.

The same two problems will be implemented in every candidate sprint (S2..S7), so the implementations here also serve as **reference NumPy answers** (the fp64 NumPy version already in `src/gpuwrf/fixtures/analytic.py` from M1) and as a **complexity baseline** for the maintainability narrative.

### Problem 1 — Stencil (3D advection-diffusion)
- Consumes M1 fixture `analytic-stencil-3d-advdiff-v1` (32×16×8 grid, fp64 reference).
- One timestep update of `phi` given the same face-velocity fields and the same scheme (4th-order horizontal advection + 2nd-order vertical + scalar diffusion) as the analytic generator at `src/gpuwrf/fixtures/analytic.py`.
- Kernel must use shared-memory tiling for the stencil halo. Acceptable tile shape e.g. 16×8×Nk with a 2-cell halo.

### Problem 2 — Column (register-heavy thermo)
- Consumes M1 fixture `analytic-column-thermo-v1` (40-level column, fp64 reference).
- One timestep update of `temperature` and `qv` given the same analytic source (moist-static-energy-preserving operator) as the analytic generator.
- Kernel must be column-resident: every cell's vertical column is processed by a single block, with thread-local prognostics in registers and intermediate state in shared memory.

## Non-Goals

- No JAX, Triton, Kokkos, CuPy, gt4py. Those are S3..S7.
- No real WRF physics. Problems 1+2 are M1 analytic shapes only.
- No multi-GPU. Single-GPU RTX 5090 only.
- No mixed precision yet — use fp64 to match M1 reference. (Mixed precision is an M4+ concern; ADR-001 only needs same-precision profiles for comparison.)
- No file changes outside the cuda_tile artifacts paths.

## File Ownership

Worker may create or edit only these paths:

- `src/gpuwrf/backends/cuda_tile/__init__.py` (new if missing — package marker)
- `src/gpuwrf/backends/cuda_tile/stencil.cu` (new — Problem 1 kernel)
- `src/gpuwrf/backends/cuda_tile/column.cu` (new — Problem 2 kernel)
- `src/gpuwrf/backends/cuda_tile/host.cpp` (new — host driver: ingest M1 fixture .npz via cnpy or hand-rolled NumPy npz reader, copy H2D, launch kernel, copy D2H, write candidate .npz)
- `src/gpuwrf/backends/cuda_tile/CMakeLists.txt` OR `src/gpuwrf/backends/cuda_tile/Makefile` (new — single-command build)
- `src/gpuwrf/backends/cuda_tile/build.sh` (new — wraps the build for portability)
- `scripts/m2_run_cuda_tile.sh` (new — full pipeline: source env, build, run stencil, run column, parse ncu output, write JSON artifacts)
- `artifacts/m2/cuda_tile/stencil_profile.json` (new — schema below)
- `artifacts/m2/cuda_tile/column_profile.json` (new)
- `artifacts/m2/cuda_tile/correctness.json` (new — `compare_fixture` round-trip result for both problems)
- `artifacts/m2/cuda_tile/maintainability.md` (new — ≤300 words)
- `artifacts/m2/cuda_tile/agent_success.json` (new — sprint_count=1 at worker close, reviewer_rejections=0 if Accept first pass, escalation_events=0)
- `tests/test_m2_cuda_tile.py` (new — at minimum: build succeeds, stencil correctness within tolerance, column correctness within tolerance, profile JSON validates schema)
- `pyproject.toml` (edit only if a Python-side JSON-parsing dep is needed; explain in worker report)

Any change outside this list requires manager approval.

## Inputs

- M1 fixtures on main:
  - `fixtures/manifests/analytic-stencil-3d-advdiff-v1.yaml` + `fixtures/samples/analytic-stencil-3d-advdiff-v1.npz`
  - `fixtures/manifests/analytic-column-thermo-v1.yaml` + `fixtures/samples/analytic-column-thermo-v1.npz`
- M1 CLI: `src/gpuwrf/validation/compare_fixture.py` — used to verify correctness.
- Toolchain (from M2-S1 scout, project memory `project_target_hardware.md`):
  - Source `/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/env_wrf_gpu.sh` for NVHPC paths.
  - Compile with `nvcc -arch=sm_120 -O3 -std=c++17`.
- `PROJECT_PLAN.md §5` (bakeoff candidate F definition).
- `PERFORMANCE_TARGETS.md` (profile JSON schema).
- `.agent/goals/M2-DONE.md §C` (candidate-coverage rules).

## Acceptance Criteria

All must hold for closeout.

### Build
1. `bash src/gpuwrf/backends/cuda_tile/build.sh` exits 0 and produces a single executable `data/scratch/cuda_tile/bench` (or equivalent path; gitignored).
2. Build uses `-arch=sm_120` (Blackwell) — verify by `cuobjdump --dump-sass data/scratch/cuda_tile/bench | head -1` showing `sm_120`.

### Correctness (Tier-1, both problems)
3. Stencil: `python -m gpuwrf.validation.compare_fixture --manifest fixtures/manifests/analytic-stencil-3d-advdiff-v1.yaml --candidate data/scratch/cuda_tile/stencil_out.npz --reference fixtures/samples/analytic-stencil-3d-advdiff-v1.npz` returns `pass: true`.
4. Column: same with `analytic-column-thermo-v1`.

### Profile JSON (per problem)
5. `artifacts/m2/cuda_tile/stencil_profile.json` and `column_profile.json` each match the `PERFORMANCE_TARGETS.md` schema:
   ```json
   {
     "benchmark": "m2_stencil|m2_column",
     "backend": "cuda-tile",
     "hardware": "RTX 5090 32GB",
     "case": "<M1 fixture_id>",
     "wall_time_s": float,
     "kernel_launches": int,
     "host_device_transfer_bytes": int,
     "occupancy_pct": float,
     "registers_per_thread": int,
     "local_memory_bytes": int,
     "achieved_bandwidth_gbps": float,
     "artifact_paths": ["<paths to nsys/ncu reports>"]
   }
   ```
6. `kernel_launches` must be ≥1 and represent the actual fused kernel count for the timestep (not a single-thread-of-execution count).
7. `local_memory_bytes` must be **0** for the column kernel (the whole point of the tile-resident design is to avoid register spilling). If non-zero, worker logs the spill in `maintainability.md` and the reviewer must explicitly accept it.
8. `host_device_transfer_bytes` is recorded — this is the per-run H2D + D2H. Not a violation for the bakeoff because we don't have a resident-state model yet.

### Profiler evidence
9. ncu (`ncu --set=full --export ...`) is invoked once per problem; the report file (`*.ncu-rep`, gitignored, lives at `data/profiler_artifacts/cuda_tile/`) is referenced from the profile JSON's `artifact_paths`.
10. The JSON's numbers are **parsed from** the ncu report, not from a hand-edited estimate. The script `scripts/m2_run_cuda_tile.sh` does the parsing.

### Maintainability narrative (≤300 words)
11. `artifacts/m2/cuda_tile/maintainability.md` covers, in this order: (a) build complexity (lines of CMake/build.sh, number of system deps), (b) error legibility on a deliberate bug (worker introduces one, captures the nvcc error, reverts), (c) debugger story (`cuda-gdb` works? `compute-sanitizer`?), (d) agent-iteration friction (how many compile-and-test cycles did the worker actually do).

### Agent-success log
12. `artifacts/m2/cuda_tile/agent_success.json`: `{sprint_count: 1, reviewer_rejections: 0 (worker initial estimate; manager fills final), escalation_events: 0, build_attempts: <count>, runtime_failures: <count>}`.

### Test suite
13. `tests/test_m2_cuda_tile.py` asserts: schema of both profile JSONs, correctness JSON's `pass: true`, file existence of ncu reports referenced.
14. `pytest -q` passes overall.

### Hygiene
15. `python scripts/validate_agentos.py` ok.
16. `python scripts/check_m1_done.py` ok (no regression).
17. No file >100 KB committed beyond pre-existing PDFs. Binary profiler artifacts live at `data/profiler_artifacts/cuda_tile/` (gitignored).

## Validation Commands

```bash
source /home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/env_wrf_gpu.sh
bash src/gpuwrf/backends/cuda_tile/build.sh
cuobjdump --dump-sass data/scratch/cuda_tile/bench | head -1
bash scripts/m2_run_cuda_tile.sh                     # idempotent: builds, runs, profiles, writes JSON
python -m gpuwrf.validation.compare_fixture --manifest fixtures/manifests/analytic-stencil-3d-advdiff-v1.yaml --candidate data/scratch/cuda_tile/stencil_out.npz --reference fixtures/samples/analytic-stencil-3d-advdiff-v1.npz
python -m gpuwrf.validation.compare_fixture --manifest fixtures/manifests/analytic-column-thermo-v1.yaml --candidate data/scratch/cuda_tile/column_out.npz --reference fixtures/samples/analytic-column-thermo-v1.npz
python -m json.tool artifacts/m2/cuda_tile/stencil_profile.json
python -m json.tool artifacts/m2/cuda_tile/column_profile.json
pytest -q
python scripts/check_m1_done.py
python scripts/check_m2_done.py    # cuda_tile candidate row should show as satisfied
git ls-files -z | xargs -0 stat -c '%s %n' | sort -nr | head -5
```

## Performance Metrics

Captured in profile JSONs per §Acceptance Criteria. The numbers themselves are **not** a sprint pass/fail (this sprint just establishes the cuda_tile baseline); ADR-001 in M2-S8 compares across candidates.

That said, sanity bounds for the reviewer to flag if violated:
- `wall_time_s` per problem ≤ 5 s on the small M1 fixtures. Anything higher suggests a launch loop or transfer pathology.
- `kernel_launches` per problem ≤ 10. The whole point of tile-resident is to fuse.
- `occupancy_pct` ≥ 25 for stencil, ≥ 20 for column. Below = something's wrong with the tile choice.
- `registers_per_thread` ≤ 64 for stencil, ≤ 128 for column (column tolerates more because of the prognostic count). Hitting 255 means the compiler is spilling.

## Proof Object

- Diff (limited to File Ownership paths).
- The 5 artifacts under `artifacts/m2/cuda_tile/`.
- ncu reports under `data/profiler_artifacts/cuda_tile/` (referenced from JSONs; not committed).
- worker / tester / reviewer / closeout / memory-patch reports.

## Risks

- **NVHPC env may not be source-able in the codex sandbox.** Fallback: hardcode the relevant paths in `build.sh` (CUDA_HOME, PATH). Worker documents.
- **ncu may require `--target-processes=all` or root.** Worker tries unprivileged first; if profiler counters are restricted, write a workaround using nsight-systems' kernel summary as a fallback, or document the limitation and emit best-effort numbers with a `profiler_limitation` field in the JSON.
- **Stencil halo bug class.** Easy to off-by-one shared-memory tile loads. The correctness test against the M1 reference is the guard.
- **Column register spilling.** If the kernel hits 255 registers (the previous wrf_gpu attempt's main pathology), the worker MUST refactor before claiming `pass`. The contract's AC #7 explicitly requires `local_memory_bytes: 0`.
- **`cnpy` is small and easy** — worker can vendor it or hand-roll the npz reader. Either is acceptable; document in `maintainability.md`.

## Handoff Requirements

- Worker pushes to branch `worker/gpt/m2-cuda-tile-stencil-column`.
- After reviewer Accept, manager merges into main, then opens M2-S3 (`cupy_or_numba`).
- Tester is **Claude Opus 4.7 xhigh** (cross-AI verification): re-runs the pipeline from a clean shell, spot-checks that the profile JSON numbers actually come from the ncu report (not fabricated), checks the kernel really compiles for sm_120 (not silently falling back to a lower CC).
