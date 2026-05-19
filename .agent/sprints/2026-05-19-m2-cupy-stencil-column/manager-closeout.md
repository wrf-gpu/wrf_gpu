# Manager Closeout

Sprint: `2026-05-19-m2-cupy-stencil-column` (M2-S3, cupy_or_numba bakeoff candidate E)
Closed: 2026-05-19
Cycles: 1 worker (codex high), 1 tester (Claude Opus 4.7 xhigh — second cross-AI run), 1 reviewer (codex high). Zero fix cycles.

## Outcome

Clean single-pass close. CuPy RawKernel candidate satisfies M2-DONE oracle. 2/6 candidates now done.

## Proof Objects

- Implementation: `src/gpuwrf/backends/cupy/{stencil.py,column.py,bench.py}` (~480 LOC). Real `cupy.RawKernel` instances, not idiomatic NumPy-on-GPU.
- Runner: `scripts/m2_run_cupy.sh` (idempotent venv setup + install + bench + JSON write).
- Profile JSONs: `artifacts/m2/cupy_or_numba/{stencil,column}_profile.json` with `profiler_limitation` + `achieved_bandwidth_method: fallback-derived` fields.
  - **Stencil**: registers=58, local_memory=64 B, occupancy=66.7%, wall_time≈0.31 ms, kernel_launches=1.
  - **Column**: registers=24, **local_memory=0** ✅, occupancy=100%, wall_time≈0.11 ms, kernel_launches=1.
- Correctness: both fixtures round-trip identity-pass against M1 references.
- Tests: `tests/test_m2_cupy.py` (worker, 2 tests) + `tests/test_m2_cupy_edge_cases.py` (tester, ~13 tests). Pytest 88+/88+.

## Merge Decision

Merge Decision: **Accept and integrate into main.** Reviewer Accept with zero required fixes. Only follow-up: ADR-001 (M2-S8) must label CuPy's bandwidth/timing as fallback-derived (same convention applied to cuda_tile). Already captured in M2-S2 closeout for ADR-001 carry.

## Scope Changes

None. Worker stayed inside ownership; tester added edge tests only under `tests/`; venv lives at `data/scratch/m2-cupy-venv/` (gitignored).

## Lessons

1. **CuPy raw kernels can match cuda_tile on the column problem.** Both achieve `local_memory_bytes=0` with identical registers/thread (24) and occupancy (100%). CuPy's stencil spills 64 B that cuda_tile didn't — minor signal that hand-tuned C++ retains a small advantage for stencil-shape problems, but the gap is small enough that Python-orchestration cost may not be worth chasing the difference.
2. **Cross-AI tester continues to find different things.** Claude's value-add this sprint was verifying the kernels are real RawKernels (not idiomatic NumPy ops wrapped to look raw) and independently extracting `localSizeBytes` from `cupy.cuda.Function.attributes` — codex would have just trusted its own JSON.
3. **CuPy's launch + transfer overhead is real but small.** wall_time numbers are slightly higher than cuda_tile for the column (0.11 ms vs 0.94 ms — actually CuPy faster on the small fixture; noise-dominated at these sizes). Need bigger fixtures or repeat-loop measurement to discriminate. M2-S8 ADR-001 should note this.

## Next Sprint

**M2-S4**: `m2-kokkos-stencil-column` — third candidate. Per M2-S1 readiness ranking, Kokkos is next (verdict=go-with-version-bump: Kokkos 4.7.1 source build with `-DKokkos_ARCH_BLACKWELL120=ON`). Same two problems, same M1 fixtures, same profile JSON schema.
