# Manager Closeout

Sprint: `2026-05-19-m2-cuda-tile-stencil-column` (M2-S2, cuda_tile bakeoff candidate F)
Closed: 2026-05-19
Cycles: 1 worker (codex high), 1 tester (Claude Opus 4.7 xhigh — added 21 honesty tests), 1 reviewer (codex high). Zero fix cycles for the implementation itself; reviewer required 3 hygiene fixes — all addressed by this closeout.

## Outcome

cuda_tile candidate F satisfied per M2-DONE oracle. Both bakeoff problems run on RTX 5090 with sm_120 SASS, both correctness pass, profile JSONs honest about ncu permission limitation. The 21 tester-added tests catch any future regression of the "numbers must come from real tools" invariant.

## Proof Objects

- Kernels: `src/gpuwrf/backends/cuda_tile/{stencil.cu,column.cu,host.cpp}` (~1100 LOC).
- Build: `Makefile`, `build.sh` (idempotent; falls back nvcc→nvc++-cuda when CUDA 13.1 + GCC 15 trip the rsqrt header bug).
- Runner: `scripts/m2_run_cuda_tile.sh` (idempotent; builds, runs, parses cuobjdump + occupancy API, emits JSON).
- Profile JSONs: `artifacts/m2/cuda_tile/{stencil,column}_profile.json` — registers/thread 58 (stencil) / 24 (column), local_memory_bytes **0** for both (the tile-resident design held), occupancy 66.7% (stencil) / 100% (column).
- Correctness: `artifacts/m2/cuda_tile/correctness.json` — both problems pass against M1 fixtures with max_abs_diff ≤ 4e-19 (column) and 0.0 (stencil).
- Maintainability narrative + agent-success log.
- Tests: `tests/test_m2_cuda_tile.py` (worker) + `tests/test_m2_cuda_tile_edge_cases.py` (tester, 21 tests). Pytest: 86/86.
- Profiler logs (gitignored, external): `data/profiler_artifacts/cuda_tile/` — ncu stdout/stderr/exit captures showing the `ERR_NVGPUCTRPERM` and fallback path.

## Merge Decision

Merge Decision: **Accept and integrate into main**, after addressing the 3 reviewer-required fixes:

1. ✅ **dispatch_role.sh extracted from sprint diff.** That commit (d4d1b30) was contaminated onto the worker branch by the manager (shared-worktree issue). Resolution: cherry-picked the fix onto main as commit `058e872`, then `git rebase --onto 6dc4c8c d4d1b30` rewrote the reviewer branch to drop d4d1b30. The reviewer-branch head is now `548ad7b` with only sprint-scoped commits.
2. ⏳ **Profiler-counter waiver for cuda_tile** — *granted here by the manager:* this run enters ADR-001's candidate matrix with the documented `profiler_limitation` field, no `.ncu-rep` required. Rationale: (a) the contract's Risks section explicitly anticipated this fallback; (b) all 6 M2 candidates will run under the same `ERR_NVGPUCTRPERM` constraint, so cross-candidate comparison is internally consistent; (c) registers/local-memory from `cuobjdump` and theoretical occupancy from the CUDA occupancy API are sufficient signals for the bakeoff's actual decision (`local_memory_bytes == 0` is the most important number, and we have it). If a future hardware-config change makes counters available, M2-S8 (ADR-001) may optionally rerun for higher-fidelity numbers, but it is not required.
3. ⏳ **Bandwidth labelling for ADR-001** — *captured here:* `achieved_bandwidth_gbps` is fallback-derived from bench-output wall_time + recorded transfer bytes, not measured by ncu performance counters. ADR-001 must present this metric with that qualifier and treat it as order-of-magnitude only, not absolute. Manager will carry this into M2-S8's contract.

## Scope Changes

None for the implementation. One process change: the contamination of `scripts/dispatch_role.sh` onto the worker branch — surfaced a real shared-worktree issue between the manager process and codex workers. Mitigation for future sprints: manager avoids committing while a worker is running, OR uses `git worktree add` for isolation. Documented as a backlog item for the M2 runbook update (handled in S3's contract or as a separate hygiene commit).

## Lessons

1. **Cross-AI tester continues to deliver real value.** Claude added 21 tests specifically targeting "did the worker fabricate numbers." A same-AI codex tester would not naturally write tests against its own honesty.
2. **`local_memory_bytes: 0` is the critical metric** — the contract's AC#7 required it, the cuda_tile candidate hit it, and that single number is the strongest predictor of how this candidate will fare against MYNN/Thompson-shape kernels at M5+.
3. **nvcc 13.1 has a real header-clash bug** with GCC 15's libstdc++ (`rsqrt` exception-specification mismatch). All M2 candidates that go through nvcc need to know this; `nvc++ -cuda` is the documented workaround.
4. **`ERR_NVGPUCTRPERM` is a deployment constraint, not a per-candidate issue.** Every M2 candidate will hit it. The contract's risk-fallback pattern (record `profiler_limitation`, derive what you can from `cuobjdump` + occupancy API) is the right pattern; all subsequent candidate sprints should follow.

## Next Sprint

**M2-S3**: `m2-cupy-stencil-column` — implement the two bakeoff problems in Python with CuPy raw CUDA kernels (`cupy.RawKernel`). Next on the readiness ladder (verdict=go, install command `pip install cupy-cuda13x==14.0.1`). Same two problems, same M1 fixtures, same profile JSON schema. Manager opens immediately after merging this S2 to main.
