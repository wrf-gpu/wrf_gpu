# Worker Report

This sprint's "worker" is the manager (per contract: M2-S8 is a decision sprint, not delegated to codex). This file is the manager's self-report of evidence consumed and reasoning chain, satisfying the lifecycle template + close_sprint size/token requirements.

## Summary

Summary: Wrote `.agent/decisions/ADR-001-backend-selection.md` (~15 KB) selecting JAX as the primary v0 backend for the GPU-native NWP rewrite, with a per-scheme gated Triton fallback option for any M5 physics scheme that exhibits register-spilling on real workloads. Spawned Codex `gpt-5.5 xhigh` critical-review and applied all 5 substantive findings. Then dispatched codex reviewer for binding judgment; applied all 6 of its findings (3 blockers + 2 majors + 1 minor) in this revision pass. ADR-001 is now `proposed, pending user acknowledgement at M2 closeout`.

## Evidence consumed

- `artifacts/m2/scout/toolchain_support_matrix.json` + `toolchain_report.md`
- `artifacts/m2/cuda_tile/{stencil,column}_profile.json` + `correctness.json` + `maintainability.md` + `agent_success.json`
- `artifacts/m2/cupy_or_numba/...` (full 5-file set)
- `artifacts/m2/kokkos/...` (full 5-file set)
- `artifacts/m2/jax/...` (full 5-file set)
- `artifacts/m2/triton/...` (full 5-file set; corrected after fix-cycle for cubin-cache contamination)
- `artifacts/m2/gt4py/...` (candidate-failure artifacts, created in this sprint to satisfy oracle)
- All 6 M2 sprint folders' `manager-closeout.md` § Lessons
- `PROJECT_PLAN.md §5` (bakeoff candidate definitions), `PROJECT_CONSTITUTION.md`, `ARCHITECTURE_PRINCIPLES.md`, `PERFORMANCE_TARGETS.md`, `PRECISION_POLICY.md`
- Project memory `project_target_hardware.md` (Blackwell + toolchain pins + ncu permission limitation + nvcc/GCC 15 header bug)
- `.agent/rules/architecture-decision-policy.md`, `.agent/rules/cross-model-review-policy.md`

## Reasoning chain (top-level)

1. **All 5 implemented candidates pass correctness and achieve `local_memory_bytes=0` on the column kernel.** The previous wrf_gpu attempt's register-spilling failure mode does not materialize on the M1 analytic surrogate for ANY candidate. This is the strongest signal that the architectural premise of the v2 rewrite is sound.
2. **JAX has the lowest register count on both problems** (stencil 48, column 22). At-scale this translates directly to better occupancy as fixtures grow. JAX also achieves the highest theoretical occupancy on the stencil (83.3%).
3. **Author velocity favors JAX significantly.** First-pass agent success (zero fix cycles), Python-only, debuggable, trivial ML coupling. Triton came second with a fix-cycle; Kokkos required source build + verbose templates; cuda_tile needed manual workarounds for nvcc-13.1+GCC-15 headers.
4. **Triton vs JAX on column** (the deepthink-brief-proposed hybrid case): Triton column regs=34 vs JAX 22 (after Triton attempt-2 fix). No spill on either. Hybrid is not obviously justified — but is held as a per-scheme fallback option for M5.
5. **GT4Py is excluded by toolchain failure** (DaCe 0.10.0 / Python 3.13 SymPy break), not by benchmark loss. Documented in candidate-failure artifacts.

## Commands Run

- `bash scripts/dispatch_role.sh critical-review .agent/decisions/REVIEW-codex-ADR-001/ --reasoning xhigh` → codex Accept-with-required-fixes; 5 findings applied.
- `bash scripts/dispatch_role.sh reviewer .agent/sprints/2026-05-19-m2-adr-001-backend-selection/ --reasoning high` → codex Accept-with-required-fixes; 6 findings applied in this revision.
- `pytest -q tests/test_adr_001_structure.py` → 4 passed.
- `python scripts/check_m1_done.py` → ok=true.
- `python scripts/check_m2_done.py` → 6/6 candidates satisfied (including gt4py failure artifacts); only milestone closeout missing.

## Proof Objects

- `.agent/decisions/ADR-001-backend-selection.md` (≥15 KB, all 4 required tokens, Selected backend regex match)
- `.agent/decisions/REVIEW-codex-ADR-001.md` (cross-model review pointer)
- `.agent/decisions/REVIEW-codex-ADR-001/proposal.md` + `critical-review.md` (Codex xhigh challenge transcript)
- `tests/test_adr_001_structure.py` (4 tests, all passing)
- `artifacts/m2/gt4py/...` (candidate-failure schema, satisfying oracle)

## Risks

- ADR-001 remains *proposed* pending user explicit approval at M2 closeout. Per the constitution, M3 cannot start on the JAX assumption until the user explicitly approves. The user has reviewed the in-progress state in real time and the manager-autonomy directive delegates everything below this gate.
- M5 first real physics implementation remains the decisive test (M5 stop/go gate). M2 evidence is necessary, not sufficient.
- Real `ncu` profiler artifacts cannot be obtained until system admin sets `nvidia-driver-perfmon-allow=1`. M3/M4 follow-up action.

## Handoff

Next: codex reviewer (this is a re-review after fixes); on Accept, manager writes MILESTONE-M2-CLOSEOUT.md, flips M2 plan to Accepted, merges + pushes, stops the /loop, presents the final state to the user with an explicit approval request.

Summary: M2 bakeoff evidence supports JAX as v0 primary; ADR-001 ready for user acknowledgement; M3 dispatch-blocked on explicit human approval per constitution.
