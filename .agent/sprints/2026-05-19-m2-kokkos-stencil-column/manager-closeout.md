# Manager Closeout

Sprint: `2026-05-19-m2-kokkos-stencil-column` (M2-S4, kokkos bakeoff candidate D)
Closed: 2026-05-19
Cycles: 1 worker (codex), 1 tester (Claude Opus xhigh — 3rd cross-AI), 1 reviewer (codex). Zero fix cycles.

## Outcome

3/6 M2 candidates satisfied. Kokkos source-built from 4.7.01 tag with `BLACKWELL120` + CUDA backend; both bakeoff problems pass correctness with sm_120 SASS.

## Proof Objects

- Kernels + host: `src/gpuwrf/backends/kokkos/{stencil,column,host}.cpp` + `CMakeLists.txt` + `build.sh`.
- Runner: `scripts/m2_run_kokkos.sh` (idempotent kokkos build + bench run + JSON write).
- Profile JSONs (same schema as cuda_tile + cupy, same `profiler_limitation` + `achieved_bandwidth_method: fallback-derived` conventions):
  - Stencil: registers=**64** (at contract limit — flagged for ADR-001), local_memory=0, occupancy=66.7%, kernel_launches=1.
  - Column: registers=40, local_memory=**0** ✅, occupancy=100%, kernel_launches=1.
- Correctness: both round-trip pass.
- Tests: 32 new tests added by Claude. Pytest 146/146.

## Merge Decision

Merge Decision: **Accept and integrate into main.** Reviewer Accept with zero required fixes. One reviewer note for ADR-001 carry: pytest currently rewrites committed timing artifacts because `m2_run_kokkos.sh` is called as part of the test setup. ADR-001 should standardize "run-once-per-CI vs per-pytest" semantics across all candidates.

## Scope Changes

None.

## Lessons

1. **3-way comparison emerging:** all 3 candidates achieve `local_memory_bytes=0` on the column kernel. Register counts differ: cuda_tile/cupy 24, kokkos 40. The Kokkos abstraction adds ~50% register tax on column but stays well below the 128 limit.
2. **Stencil register pressure for Kokkos is at the contract limit (64).** This is the first signal that Kokkos's abstraction tax could matter for more complex kernels (M5 physics). Worth flagging in ADR-001's per-candidate column.
3. **Kokkos source build takes ~10–15 min one-time** (cached afterward). Acceptable for v0 but a real friction signal for the agent-success log.
4. **Wall-time numbers across candidates are noise-dominated at M1 fixture sizes** (microseconds). Cross-candidate timing comparison in ADR-001 needs either bigger fixtures or repeat-loop measurement; this is a known limitation to document.

## Next Sprint

**M2-S5**: `m2-jax-stencil-column` — JAX/XLA candidate. Per M2-S1 readiness: pin `jax[cuda13]==0.10.0`. Same two problems, same M1 fixtures, same profile JSON schema. This is the user's gut-favorite ("why not just JAX?") — ADR-001 needs honest JAX numbers, especially on the column kernel where XLA may or may not avoid register spilling.
