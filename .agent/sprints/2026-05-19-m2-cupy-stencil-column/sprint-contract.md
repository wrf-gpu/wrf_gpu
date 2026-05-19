# Sprint Contract

Sprint ID: `2026-05-19-m2-cupy-stencil-column`
Milestone: M2 — Backend Bakeoff
Sequence: S3 (per M2-S1 readiness ranking — cupy_or_numba is 2nd: verdict=go, single pip install)
Worker: gpt-kernel-worker (Codex `gpt-5.5` `high`)
Tester: sonnet-test-engineer (Claude Opus 4.7 `xhigh` — cross-AI verification)
Reviewer: opus-reviewer (Codex `gpt-5.5` `high`)
Candidate family: `cupy_or_numba` (Python with CuPy raw CUDA kernels; Numba CUDA acceptable if worker has a strong reason)
Approval status: opened 2026-05-19 by manager after M2-S2 closeout.

## Objective

Implement both bakeoff problems in **Python + CuPy raw CUDA kernels** (`cupy.RawKernel`), running against the M1 fixtures, with profile-quality evidence in the same schema as M2-S2's cuda_tile. This is the **lowest-friction Python escape hatch** — manager wants to measure whether Python orchestration + raw-CUDA inner loops can approach the cuda_tile C++ performance.

The two problems are the *same* M1-fixture-driven stencil and column as M2-S2 (definitions in `src/gpuwrf/fixtures/analytic.py`):
- **Problem 1**: 3D advection-diffusion stencil, single timestep, 32×16×8 grid, fp64.
- **Problem 2**: register-heavy thermo column, single timestep, 40-level column, fp64.

Worker MAY choose Numba CUDA instead of CuPy if there's a defensible reason (e.g. CuPy's RawKernel API doesn't support a needed feature). Decision recorded in maintainability.md.

## Non-Goals

- No JAX, Triton, Kokkos, cuda_tile, gt4py — those are other M2 sprints.
- No "use CuPy as numpy-backend" — the bakeoff specifically measures hand-written raw CUDA kernels via CuPy. Idiomatic NumPy-like CuPy is too easy and doesn't show the candidate's real ceiling.
- No mixed precision.
- No multi-GPU.

## File Ownership

Worker may create or edit only these paths:

- `src/gpuwrf/backends/cupy/__init__.py` (new if missing)
- `src/gpuwrf/backends/cupy/stencil.py` (new — Problem 1 raw-CUDA kernel + Python host driver)
- `src/gpuwrf/backends/cupy/column.py` (new — Problem 2 raw-CUDA kernel + host driver)
- `src/gpuwrf/backends/cupy/bench.py` (new — CLI that reads M1 fixture, runs both kernels, writes candidate outputs + raw profile data)
- `scripts/m2_run_cupy.sh` (new — venv setup at `data/scratch/m2-cupy-venv/`, install `cupy-cuda13x==14.0.1` per M2-S1 scout pin, run bench, parse to JSON)
- `artifacts/m2/cupy_or_numba/stencil_profile.json` (new)
- `artifacts/m2/cupy_or_numba/column_profile.json` (new)
- `artifacts/m2/cupy_or_numba/correctness.json` (new)
- `artifacts/m2/cupy_or_numba/maintainability.md` (new, ≤300 words)
- `artifacts/m2/cupy_or_numba/agent_success.json` (new)
- `tests/test_m2_cupy.py` (new — build/import succeeds, correctness pass, profile JSON validates)
- `pyproject.toml` (do NOT add cupy as a project dep — keep it in the sprint venv only)

Anything outside this list requires manager approval.

## Inputs

- M1 fixtures on main (analytic-stencil + analytic-column manifests + samples).
- M1 CLI `src/gpuwrf/validation/compare_fixture.py`.
- M2-S1 scout pin: `cupy-cuda13x==14.0.1`.
- M2-S2 cuda_tile profile JSONs as schema reference + "what good looks like" target.
- Project memory `project_target_hardware.md` (note the ncu permission limitation and the `profiler_limitation` fallback pattern — applies here too).

## Acceptance Criteria

All must hold.

### Install & smoke
1. `bash scripts/m2_run_cupy.sh` creates `data/scratch/m2-cupy-venv/`, pip-installs `cupy-cuda13x==14.0.1` (and only needed transitives), runs `python -c "import cupy; print(cupy.cuda.runtime.runtimeGetVersion())"` and prints a CUDA 13 runtime version.
2. The script is idempotent: a second run reuses the venv, doesn't re-install, runs the bench again.

### Correctness
3. Stencil: `compare_fixture` round-trip identity-pass.
4. Column: `compare_fixture` round-trip identity-pass.

### Profile JSON (per problem)
5. Both `stencil_profile.json` and `column_profile.json` match the `PERFORMANCE_TARGETS.md` schema and include the `profiler_limitation` field (per the M2-S2 fallback pattern). Required numeric fields:
   - `wall_time_s` — measured by host-side `time.perf_counter_ns` around the kernel launch + synchronize.
   - `kernel_launches` — count of `cupy.RawKernel` invocations per problem (target ≤ 5).
   - `host_device_transfer_bytes` — sum of `cupy.asarray` H2D + `cupy.asnumpy` D2H bytes per run.
   - `occupancy_pct` — from `cupy.cuda.Function.occupancy()` or `cuOccupancyMaxActiveBlocksPerMultiprocessor` ctypes call.
   - `registers_per_thread` — from compiled-PTX inspection via `cuobjdump --dump-sass` on the dumped kernel binary, OR from `cupy.cuda.Function.attributes['numRegs']`.
   - `local_memory_bytes` — from `cupy.cuda.Function.attributes['localSizeBytes']`. **Must be 0 for the column kernel** (matches the M2-S2 cuda_tile requirement).
   - `achieved_bandwidth_gbps` — derived `host_device_transfer_bytes / wall_time_s / 1e9`; label as `fallback-derived` (same convention as M2-S2).

### Maintainability narrative (≤300 words)
6. Covers: (a) install complexity (pip install + venv lines), (b) error legibility (worker introduces a deliberate kernel bug, captures the CuPy error message, reverts), (c) debugger story (`cuda-gdb` works on RawKernels? `cupy.cuda.profiler`?), (d) agent-iteration friction.

### Agent-success
7. `agent_success.json` populated.

### Tests
8. `tests/test_m2_cupy.py`: schema validation of both profile JSONs, correctness JSON pass-check, file existence of artifact paths.
9. `pytest -q` passes overall.

### Hygiene
10. `validate_agentos.py` ok.
11. `check_m1_done.py` ok (no regression).
12. No file >100 KB committed beyond pre-existing.
13. `local_memory_bytes` for column kernel = 0 (or worker writes a justified note in maintainability.md and reviewer must explicitly accept).

## Validation Commands

```bash
bash scripts/m2_run_cupy.sh                                # idempotent: venv + install + run + JSON write
python -m gpuwrf.validation.compare_fixture --manifest fixtures/manifests/analytic-stencil-3d-advdiff-v1.yaml --candidate data/scratch/m2-cupy/stencil_out.npz --reference fixtures/samples/analytic-stencil-3d-advdiff-v1.npz
python -m gpuwrf.validation.compare_fixture --manifest fixtures/manifests/analytic-column-thermo-v1.yaml --candidate data/scratch/m2-cupy/column_out.npz --reference fixtures/samples/analytic-column-thermo-v1.npz
python -m json.tool artifacts/m2/cupy_or_numba/stencil_profile.json
python -m json.tool artifacts/m2/cupy_or_numba/column_profile.json
pytest -q
python scripts/check_m1_done.py
python scripts/check_m2_done.py    # cupy_or_numba candidate row should show satisfied
git ls-files -z | xargs -0 stat -c '%s %n' | sort -nr | head -5
```

## Performance Metrics

Captured in profile JSONs. Sanity bounds for reviewer:
- `wall_time_s` per problem ≤ 5 s on M1 fixtures.
- `kernel_launches` per problem ≤ 5 (raw kernels should fuse the work; if it's 50, the kernel is too granular).
- `registers_per_thread` ≤ 64 stencil, ≤ 128 column. Hitting 255 = compiler is spilling.
- **`local_memory_bytes` for column = 0**. The whole reason this candidate is in the bakeoff is to measure whether Python orchestration + raw CUDA can defeat register spilling on column kernels.

## Proof Object

- Diff (File Ownership only).
- 5 artifacts under `artifacts/m2/cupy_or_numba/`.
- Lifecycle reports.

## Risks

- **CuPy RawKernel API may not expose all the `cuOccupancy*` calls cleanly.** Workaround: call via `ctypes.CDLL("libcuda.so.1")` and feed `cupy.cuda.Function.handle`. Worker documents.
- **`cupy.cuda.Function.attributes`** is the right place to get registers + local-memory. If those fields aren't populated for RawKernels, worker falls back to `cuobjdump` on the cubin dumped via `cupy.cuda.Module.dump`.
- **Venv install size** can be ~3 GB for CuPy CUDA13. Worker monitors `df -h /mnt/data`; writes BLOCKER if below 50 GB free.
- **Numba CUDA fallback** — only if CuPy can't be made to work; worker justifies in maintainability.md.

## Handoff Requirements

- Worker pushes to branch `worker/gpt/m2-cupy-stencil-column`.
- Tester is Claude Opus 4.7 xhigh (cross-AI). Tester should specifically verify: (a) the kernels are real raw-CUDA, not idiomatic CuPy NumPy ops; (b) `local_memory_bytes` claim is real (read it from `Function.attributes` independently); (c) the venv install actually pinned to 14.0.1.
- After reviewer Accept, manager merges to main and opens M2-S4 (kokkos).

## Note on manager-during-worker commit hygiene

Per memory `feedback_manager_autonomy.md` "Operational addition 2026-05-19": manager will NOT commit to other files while this worker is in flight. Any manager-side script tweaks will wait for the tester/reviewer cycle.
