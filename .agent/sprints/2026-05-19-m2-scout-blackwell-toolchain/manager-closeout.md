# Manager Closeout

Sprint: `2026-05-19-m2-scout-blackwell-toolchain` (M2-S1, Blackwell readiness scout)
Closed: 2026-05-19
Cycles: 1 worker (codex), 1 tester (Claude Opus 4.7 xhigh — **first cross-AI verification**), 1 reviewer (codex). Zero fix cycles.

## Outcome

Clean single-pass close. Cross-AI verification machinery validated end-to-end: Claude tester independently re-ran the scout, cross-checked all 6 version pins against actually-installed venvs, reproduced the gt4py block. Codex reviewer Accept.

## Proof Objects

- `artifacts/m2/scout/toolchain_support_matrix.json` — 6 candidates, hardware metadata, verdicts, version pins, install commands.
- `artifacts/m2/scout/toolchain_report.md` — ≤2000-word narrative with readiness ordering recommendation.
- `artifacts/m2/scout/hello_gpu/<candidate>/` — runnable hello-GPU programs + outputs + exit codes for 5 non-blocked candidates.
- `scripts/m2_scout_hello_gpu.sh` — idempotent re-runner.
- `tests/test_m2_scout_matrix.py` + `tests/test_m2_scout_matrix_extras.py` (16 Claude-added edge tests). Pytest: 63/63.

### Readiness ordering (for S2..S7 dispatch)

| Rank | Candidate | Verdict | Install |
|---|---|---|---|
| 1 | `cuda_tile` | go | CUDA 13.1 / NVHPC 26.3, `nvcc -arch=sm_120` (already on system) |
| 2 | `cupy_or_numba` | go | `pip install cupy-cuda13x==14.0.1` |
| 3 | `kokkos` | go-with-version-bump | Kokkos 4.7.1 source build, `-DKokkos_ARCH_BLACKWELL120=ON` |
| 4 | `jax` | go-with-version-bump | `pip install jax[cuda13]==0.10.0` |
| 5 | `triton` | go-with-version-bump | `triton==3.7.0` + `torch==2.12.0` (CUDA13) |
| 6 | `gt4py` | **blocked** | DaCe 0.10.0 fails under Python 3.13 (SymPy break). Future scout may try Python 3.12 venv. |

## Merge Decision

Merge Decision: **Accept and integrate into main**. Worker branch `worker/gpt/m2-scout-blackwell-toolchain` carries the full sprint (worker + tester + reviewer commits via reviewer branch).

## Scope Changes

None. Stayed inside contracted scope. `data/scratch/m2-scout-venv/` created for per-candidate sandboxes (gitignored). gt4py's `blocked` verdict is a contracted outcome, not a scope failure.

## Lessons

1. **Cross-AI tester catches different things.** Claude Opus's value-add this sprint: 16 new edge-case tests probing the scout matrix schema, and a real cross-check of version pins against the actually-installed venvs (codex worker had pinned correctly, but the verification was independent). Future contracts should leverage Claude's "try to break it" instinct more explicitly in tester AC.
2. **gt4py block is a Python-version mismatch, not a Blackwell mismatch.** A remediation scout under Python 3.12 might recover it. Manager defers this to post-S7 unless ADR-001 evidence shows we materially need a stencil DSL beyond Triton/CUDA.
3. **The readiness ordering correctly puts low-friction candidates first.** S2 (cuda_tile) requires zero install — straight `nvcc` with existing NVHPC.

## Next Sprint

**M2-S2**: `m2-cuda-tile-stencil-column` — implement the two bakeoff problems (3D advection-diffusion stencil + register-heavy column kernel) in explicit CUDA C++ with shared-memory tile-resident kernels. Produces `artifacts/m2/cuda_tile/{stencil,column}_profile.json` + `correctness.json` + `maintainability.md` + `agent_success.json`. Manager opens immediately after this closeout.
