# Worker Report

## Summary
Summary: Implemented the M2 OpenAI Triton candidate for both required bakeoff problems: a direct `@triton.jit` 3D stencil kernel and a direct `@triton.jit` 40-level thermo-column kernel. The runner creates/reuses `data/scratch/m2-triton-venv/`, installs the pinned `triton==3.7.0` and `torch==2.12.0` stack plus NumPy for NPZ fixture I/O, runs Nsight Compute capture attempts, writes candidate NPZs, compares both fixtures, and emits the M2 profile/correctness artifacts. Stencil parity is exact. Column parity passes after forcing scalar constants through fp64 Triton expressions. Column `local_memory_bytes` is 0.

## Files Changed
- `src/gpuwrf/backends/triton/__init__.py`
- `src/gpuwrf/backends/triton/stencil.py`
- `src/gpuwrf/backends/triton/column.py`
- `src/gpuwrf/backends/triton/bench.py`
- `scripts/m2_run_triton.sh`
- `tests/test_m2_triton.py`
- `artifacts/m2/triton/stencil_profile.json`
- `artifacts/m2/triton/column_profile.json`
- `artifacts/m2/triton/correctness.json`
- `artifacts/m2/triton/maintainability.md`
- `artifacts/m2/triton/agent_success.json`
- `.agent/sprints/2026-05-19-m2-triton-stencil-column/worker-report.md`

## Commands Run
- `bash scripts/m2_run_triton.sh`
  - exit 0
  - stdout: `3.7.0 13.0`
  - stderr: empty
- `python -m gpuwrf.validation.compare_fixture --manifest fixtures/manifests/analytic-stencil-3d-advdiff-v1.yaml --candidate data/scratch/m2-triton/stencil_out.npz --reference fixtures/samples/analytic-stencil-3d-advdiff-v1.npz`
  - exit 0
  - output: JSON `pass: true`; `phi_next max_abs_diff: 0.0`; no first failure.
- `python -m gpuwrf.validation.compare_fixture --manifest fixtures/manifests/analytic-column-thermo-v1.yaml --candidate data/scratch/m2-triton/column_out.npz --reference fixtures/samples/analytic-column-thermo-v1.npz`
  - exit 0
  - output: JSON `pass: true`; `temperature_next max_abs_diff: 0.0`; `mse_delta max_abs_diff: 1.0802470029602773e-12`; no first failure.
- `python -m json.tool artifacts/m2/triton/stencil_profile.json`
  - exit 0
  - output: valid JSON; `backend: triton`; `kernel_launches: 1`; `registers_per_thread: 60`; `local_memory_bytes: 0`; `occupancy_pct: 66.66666666666667`; `wall_time_s: 2.676e-05`; `achieved_bandwidth_method: fallback-derived`.
- `python -m json.tool artifacts/m2/triton/column_profile.json`
  - exit 0
  - output: valid JSON; `backend: triton`; `kernel_launches: 1`; `registers_per_thread: 60`; `local_memory_bytes: 0`; `occupancy_pct: 70.83333333333333`; `wall_time_s: 1.519e-05`; `achieved_bandwidth_method: fallback-derived`.
- `pytest -q`
  - exit 0
  - output: `189 passed in 90.93s (0:01:30)`
- `python scripts/check_m1_done.py`
  - exit 0
  - output: `{"errors": [], "manifest_dir": "fixtures/manifests", "ok": true, "sprints_closed": 3}`
- `python scripts/validate_agentos.py`
  - exit 0
  - output: `{"errors": [], "ok": true, "required_files_checked": 31, "skills_checked": 13}`
- `python scripts/check_m2_done.py`
  - exit 1 after this report was expanded
  - output: `candidates_satisfied: 5`, `candidates_total: 6`, `ok: false`; remaining errors are reviewer/tester/manager/memory report stubs, missing `artifacts/m2/gt4py/`, missing ADR-001, missing M2 closeout, M2 reviewer decision not accepted, and missing tester log provenance. These are outside worker file ownership.
- `git ls-files -z | xargs -0 stat -c '%s %n' | sort -nr | head -5`
  - exit 0
  - output:
    - `1540850 WRF GPU Porting_ Architecture & Verification.pdf`
    - `97515 wrf to gpu gpt5.5 deep research.pdf`
    - `61080 fixtures/samples/analytic-stencil-3d-advdiff-v1.npz`
    - `31304 tests/test_m2_jax_edge_cases.py`
    - `28015 tests/test_m2_kokkos_edge_cases.py`

## Proof Objects
- `artifacts/m2/triton/stencil_profile.json`
- `artifacts/m2/triton/column_profile.json`
- `artifacts/m2/triton/correctness.json`
- `artifacts/m2/triton/maintainability.md`
- `artifacts/m2/triton/agent_success.json`
- `data/scratch/m2-triton/stencil_out.npz`
- `data/scratch/m2-triton/column_out.npz`
- `data/profiler_artifacts/triton/stencil_cuobjdump_resource_usage.txt`
- `data/profiler_artifacts/triton/column_cuobjdump_resource_usage.txt`
- `data/profiler_artifacts/triton/stencil_ncu_stdout.txt`
- `data/profiler_artifacts/triton/stencil_ncu_stderr.txt`
- `data/profiler_artifacts/triton/stencil_ncu_exit.txt`
- `data/profiler_artifacts/triton/column_ncu_stdout.txt`
- `data/profiler_artifacts/triton/column_ncu_stderr.txt`
- `data/profiler_artifacts/triton/column_ncu_exit.txt`

## Risks
- Nsight Compute was invoked, but profile JSON uses fallback-derived timing/bandwidth and cuobjdump-derived register/local-memory metrics to match existing M2 conventions and handle local performance-counter limitations.
- Triton depends on the heavy torch runtime for launch/buffer management in this sprint; it is isolated to `data/scratch/m2-triton-venv/` and not added to `pyproject.toml`.
- `check_m2_done.py` cannot pass in this worker branch because M2 closeout, ADR-001, GT4Py evidence, tester/reviewer reports, and manager closeout are owned by other roles/sprints.

## Handoff
Objective: deliver the Triton M2 stencil+column candidate evidence.

Files changed: listed above; no governance, goal, tester, reviewer, manager closeout, or memory patch files were edited.

Commands run: listed above with outputs.

Proof objects produced: listed above; tracked M2 artifacts are under `artifacts/m2/triton/`, with profiler scratch under `data/`.

Unresolved risks: M2 as a milestone is not closeable from this worker branch alone; column occupancy/register count should be independently re-derived by tester from copied cubins/cache.

Next decision needed: tester/reviewer should verify `@triton.jit` usage, cubin resource metrics, venv-only torch dependency, and whether Triton column evidence changes ADR-001 relative to JAX.
