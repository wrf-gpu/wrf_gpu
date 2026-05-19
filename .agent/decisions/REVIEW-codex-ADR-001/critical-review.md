# Critical Review: ADR-001 Backend Selection

Decision: Accept with required fixes

## Top three structural concerns

1. The backend choice itself is defensible as a v0 primary choice, but the ADR currently treats an irreversible architecture lock as manager-exercised rather than explicitly human-approved.
2. The M2 evidence set is not yet closed: GT4Py is treated as skipped in the proposal, while the milestone oracle still requires a candidate directory with failure artifacts, maintainability, and agent-success records.
3. The JAX-plus-Triton fallback is too broad for a no-new-ADR clause. It changes the selected backend shape at the exact point, M5 real physics, where the current evidence is weakest.

## Findings

1. Blocker: Irreversible approval is not shown as an explicit human approval artifact. `PROJECT_CONSTITUTION.md:16` says irreversible architecture decisions require human approval, and `.agent/rules/architecture-decision-policy.md:13` repeats that rule. The proposal instead states that the manager exercises irreversible-decision authority and reports later (`.agent/decisions/REVIEW-codex-ADR-001/proposal.md:7`). Required fix: before merge, add a concrete human approval record or revise status to "proposed, pending human approval"; do not close M2 on manager-only approval.

2. Blocker: GT4Py coverage does not satisfy the objective M2 gate. `.agent/goals/M2-DONE.md:27-35` requires all six candidates to have profile/correctness artifacts or candidate-failure artifacts plus maintainability and agent-success files. `scripts/check_m2_done.py:24` includes `gt4py`, and `scripts/check_m2_done.py:81-107` requires a candidate directory and the per-candidate files. The proposal only says GT4Py was excluded by the scout (`.agent/decisions/REVIEW-codex-ADR-001/proposal.md:19`). Required fix: add `artifacts/m2/gt4py/{stencil,column}_failure.json`, `maintainability.md`, and `agent_success.json`, or formally patch the M2 gate before claiming closure.

3. Major: The pre-authorized Triton fallback bypasses the architecture-decision boundary. `.agent/rules/architecture-decision-policy.md:3-5` requires an ADR for backend selection, while the proposal allows a later JAX-to-Triton physics implementation with "no new ADR required" (`.agent/decisions/REVIEW-codex-ADR-001/proposal.md:15`). Required fix: either declare the selected backend as `hybrid:jax+triton-physics` now with an explicit integration proof plan, or keep `Selected backend: jax` and require a bounded follow-on ADR/reviewer gate before any M5 physics scheme moves to Triton.

4. Major: The profiling evidence is useful but lower fidelity than the plan originally required. `PROJECT_PLAN.md:73-76` calls for `ncu`/`nsys` JSON including occupancy, registers, local memory, bandwidth, and transfer count; `PERFORMANCE_TARGETS.md:5-7` bars GPU optimization claims without profiler artifacts. The proposal states that `ncu` was unavailable and bandwidth/occupancy are fallback-derived (`.agent/decisions/REVIEW-codex-ADR-001/proposal.md:21`). Required fix: preserve JAX as the leading evidence-based choice, but phrase all performance conclusions as fallback-profiled and micro-fixture-limited; add an explicit M3/M4 action to obtain real Nsight artifacts once perf-counter permission is fixed.

5. Major: The analytic column surrogate does not justify the strength of the M5 fallback language. The proposal correctly admits real Thompson/MYNN physics has much larger branch and variable pressure (`.agent/decisions/REVIEW-codex-ADR-001/proposal.md:78-79`), but then locks the revisit trigger so narrowly that only failure of both JAX restructuring and the Triton fallback forces a re-ADR (`.agent/decisions/REVIEW-codex-ADR-001/proposal.md:99-104`). Required fix: add an M5 stop/go proof object for the first real physics scheme before treating the fallback as exercised.

6. Minor: The review/ADR packaging does not currently match the stated contract. The role prompt requires a sprint contract at `.agent/decisions/REVIEW-codex-ADR-001/sprint-contract.md` (`.agent/decisions/REVIEW-codex-ADR-001/role-prompts/critical-review.md:11`), but that file is absent. The ADR sprint contract also requires a companion pointer at `.agent/decisions/REVIEW-codex-ADR-001.md` (`.agent/sprints/2026-05-19-m2-adr-001-backend-selection/sprint-contract.md:70-73`), and the M2 oracle checks for that pointer (`scripts/check_m2_done.py:132-134`). Required fix: add the pointer file after this review and either add the missing review-local contract or correct the prompt generator.

## Dissent

I do not dissent from JAX as the primary v0 backend on the evidence available. JAX has the cleanest register story on the two M2 fixtures, first-pass agent success, and a good fit for the Python-first/ML-coupled project shape.

I dissent from merging ADR-001 as written. The proposal overcloses the decision in three places: human approval, GT4Py coverage, and fallback authority. The right decision is "JAX primary, Triton contingency under a later proof gate," not "JAX locked with an unreviewed Triton escape hatch."

## Closing recommendation

Proceed with JAX as the selected primary backend after the required fixes above. Do not close M2 until `python scripts/check_m2_done.py` no longer reports the missing GT4Py artifacts, missing cross-model pointer, and ADR-token/closure failures. Treat the first real M5 physics implementation as the next decisive test of the JAX assumption.
