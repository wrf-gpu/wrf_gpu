# Manager Closeout

Sprint: `2026-05-19-m2-triton-stencil-column` (M2-S6, triton bakeoff candidate B)
Closed: 2026-05-19
Cycles: 2 worker attempts (attempt 1 had a cubin-cache contamination bug in resource extraction; attempt 2 fixed it), 1 tester (Claude Opus xhigh — caught the bug), 2 reviewer (Reject on attempt 1, Accept on attempt 2 from a `git worktree`-isolated codex session).

## Outcome

5/6 M2 candidates satisfied. Triton column actual numbers (after fix): regs=34, local=0, occ=100%, kernel_launches=1, wall≈0.027 ms. Stencil unchanged at regs=60, local=0, occ=66.7%, 1 launch, 0.025 ms.

## Proof Objects

- Implementation: `src/gpuwrf/backends/triton/{stencil,column,bench}.py` (~600 LOC, real `@triton.jit` kernels in fp64).
- Runner: `scripts/m2_run_triton.sh` (venv with `triton==3.7.0` + `torch==2.12.0`, idempotent).
- Profile JSONs with kernel-symbol-aware cubin parsing (the attempt-2 fix).
- Correctness: both fixtures pass.
- Tests: `tests/test_m2_triton.py` + `tests/test_m2_triton_edge_cases.py` (44 tests including the cross-AI honesty checks that caught the bug). Pytest 233/233.

## Merge Decision

Merge Decision: **Accept and integrate into main.** Reviewer attempt-2 Accept with one optional cleanup (stale comment in tester edge-case file). Integrate via `git merge --no-ff reviewer/opus/m2-triton-stencil-column` into main.

## Scope Changes

None. The bug found on attempt 1 was inside the worker's File Ownership scope; fix-cycle stayed inside the same scope. Codex reviewer (attempt 2) used `git worktree add` for isolation — accidental discovery of a better isolation pattern than my current shared-worktree approach. Worth following up.

## Lessons

1. **Cross-AI tester found a bug a same-AI tester would have missed.** Claude Opus's `test_column_profile_registers_match_column_kernel_cubin` directly checked the *contradiction* between the bench's reported value (60) and what cuobjdump on the column kernel's own cubin actually said (34). Codex worker's bench was using `max()` across all Triton-cached cubins — when both kernels ran in one process, the column profile picked up the stencil cubin's registers. The committed cuobjdump text literally had both sections; the truth was on disk; the profile JSON just contradicted it. This is the platonic example of why the cross-AI testing pattern exists.
2. **Codex reviewer used `git worktree add` for isolation.** Solved the manager-worker shared-worktree contamination problem at codex's end. The reviewer commit (`8157694`) landed cleanly in a sibling directory `/home/enric/src/wrf_gpu2-review-m2-triton/` and pushed back as the branch tip. Pattern worth adopting for manager-side dispatch_role.sh going forward (deferred — works as-is for now).
3. **Cache contamination is a real bench risk for Triton.** Future Triton sprints should default to either: (a) clear `TRITON_CACHE_DIR` between kernel runs in the bench, or (b) parse cubins by kernel-symbol name. Worker chose (b). Document as M2 lesson for ADR-001 risk assessment.

## ADR-001 implications (post-Triton)

5-way picture, corrected:

| | Stencil regs/local/occ/wall | Column regs/local/occ/wall |
|---|---|---|
| cuda_tile | 58 / 0 / 66.7% / 0.9 ms | 24 / 0 / 100% / 1.0 ms |
| cupy | 58 / 64 / 66.7% / 0.06 ms | 24 / 0 / 100% / 0.03 ms |
| kokkos | 64 / 0 / 66.7% / 0.09 ms | 40 / 0 / 100% / 0.10 ms |
| **jax** | **48** / 0 / **83.3%** / 0.05 ms | **22** / 0 / 83.3% / 0.05 ms |
| triton | 60 / 0 / 66.7% / 0.03 ms | 34 / 0 / 100% / 0.03 ms |

**No candidate spills local memory on column.** The JAX-Achilles-heel concern (XLA register spilling on column physics) does NOT materialize on the analytic surrogate. Real M5 physics (Thompson, MYNN) is still the open question.

**JAX has the lowest register count on both problems.** That's the cleanest signal.

**Hybrid (JAX + Triton) is not obviously justified** by this evidence. Pure JAX is the leading ADR-001 candidate.

## Next Sprint

**M2-S7**: skip gt4py remediation for v0 — 5 candidates is sufficient evidence for ADR-001, and the GT4Py Python-3.12-venv remediation could be a post-M2 task if M5 reveals a need for a stencil DSL beyond JAX. Proceed directly to **M2-S8 ADR-001 decision sprint**: manager-owned, with Codex critical-review as second opinion per the manager-autonomy directive.
