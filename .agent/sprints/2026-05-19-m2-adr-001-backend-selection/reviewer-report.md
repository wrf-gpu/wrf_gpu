# Reviewer Report — M2 ADR-001 Backend Selection

Objective: independently review ADR-001, the critical-review response, sprint proof objects, and validation oracles for sprint `2026-05-19-m2-adr-001-backend-selection`.

## Findings

1. **blocker — irreversible approval is still implicit, not explicit human approval.** `PROJECT_CONSTITUTION.md:16` and `.agent/rules/architecture-decision-policy.md:13` require human approval for irreversible architecture decisions. The ADR says approval happens by the user reading the M2 closeout and "not objecting" (`.agent/decisions/ADR-001-backend-selection.md:5`). Non-objection is not an approval artifact. Required fix: mark ADR-001 as pending explicit human approval, or add a concrete human approval record before M3 work depends on the backend lock.

2. **blocker — required ADR structural test is missing, and the ADR does not meet the stated line format.** The sprint file ownership and acceptance criteria require `tests/test_adr_001_structure.py` (`.agent/sprints/2026-05-19-m2-adr-001-backend-selection/sprint-contract.md:37`, `:79`). That file is absent; `pytest -q tests/test_adr_001_structure.py` fails with "file or directory not found". The ADR uses `**Selected backend: jax**` (`.agent/decisions/ADR-001-backend-selection.md:11`), not a plain line matching the required regex. Required fix: add the structural test and format ADR-001 with plain `Decision:` and `Selected backend: jax` lines.

3. **blocker — lifecycle proof reports are unfinished stubs.** The contract requires standard lifecycle reports and a worker self-report documenting evidence consumed (`.agent/sprints/2026-05-19-m2-adr-001-backend-selection/sprint-contract.md:104`, `:124`). Current `worker-report.md`, `tester-report.md`, `manager-closeout.md`, and `memory-patch.md` are template stubs (`worker-report.md:1`, `tester-report.md:14`, `manager-closeout.md:1`, `memory-patch.md:14`). `python scripts/close_sprint.py ...` fails all of them as <400 bytes. Required fix: complete the required lifecycle reports, or formally reconcile the "tester not applicable" contract with the closeout oracle instead of leaving a stub.

4. **major — GT4Py failure artifacts were added outside the sprint file ownership list.** The contract allows only ADR/review files plus `tests/test_adr_001_structure.py` (`sprint-contract.md:31-39`) and explicitly says the manager may not modify candidate artifacts (`sprint-contract.md:39`). This sprint adds `artifacts/m2/gt4py/stencil_failure.json` and related files (`artifacts/m2/gt4py/stencil_failure.json:1`, `artifacts/m2/gt4py/agent_success.json:39`). The artifacts are useful and likely necessary for the M2 oracle, but the ownership breach must be recorded as a contract amendment or explicit scope exception before acceptance.

5. **major — critical-review dissent is summarized, not preserved verbatim.** The contract requires the ADR `Dissent` section to preserve the Codex critical-review disagreements verbatim or explicitly note none (`sprint-contract.md:67`). ADR-001 summarizes the three dissent points and quotes only one sentence (`.agent/decisions/ADR-001-backend-selection.md:83-90`), while the actual blocker/major findings span `.agent/decisions/REVIEW-codex-ADR-001/critical-review.md:13-21`. Required fix: include the critical review's blocker/major dissent verbatim in ADR-001 or explicitly point to an appendix treated as the verbatim record.

6. **minor — audit trail references a missing memory file.** The contract lists project memory `project_target_hardware.md` as an input (`sprint-contract.md:54`), and ADR-001 includes it in evidence files (`.agent/decisions/ADR-001-backend-selection.md:130`), but no such file exists under `.agent/`. Required fix: either add the approved memory file through the patch protocol or remove/replace the stale audit reference.

## Contract Compliance

Pass: ADR-001 exists and is >2000 bytes; the critical review, proposal, and pointer file exist; the backend selection of JAX is evidence-based enough for a proposed v0 primary backend; the ADR now acknowledges fallback-derived profile fidelity and an M5 stop/go gate.

Fail: structural test missing; lifecycle reports incomplete; closeout oracle fails; explicit human approval is not present; file ownership was exceeded for GT4Py artifacts; ADR formatting does not satisfy the required selected-backend regex.

## Correctness Risks

The JAX choice is reasonable on M2 evidence, but the ADR should remain proposed until explicit approval and the M5 real-physics gate. The analytic column fixture does not prove Thompson/MYNN behavior, and ADR-001 correctly treats that as a future stop/go point.

## Performance Risks

All profile evidence is fallback-derived because Nsight performance counters are unavailable. This is acceptable for backend selection only if future M3/M4 performance claims are blocked on real profiler artifacts. Bench/test reruns also mutate tracked `artifacts/m2/*_profile.json`, so profile artifacts need a cleaner immutable/provenance story before final closeout.

## Commands Run

- `python scripts/check_m1_done.py` → pass on rerun.
- `python scripts/check_m2_done.py` → fail: current sprint not closed, missing M2 closeout, milestone reviewer decision not accepted, tester provenance missing; candidates_satisfied=6/6.
- `python scripts/close_sprint.py .agent/sprints/2026-05-19-m2-adr-001-backend-selection` → fail on stub reports.
- `pytest -q tests/test_adr_001_structure.py` → fail, file missing.
- `pytest -q` → 233 passed.
- `grep -E '^Decision:|^Selected backend:|^## Evidence|^## Dissent' .agent/decisions/ADR-001-backend-selection.md` → no plain `Decision:` or `Selected backend:` lines.

## Proof Objects Produced

This reviewer report only: `.agent/sprints/2026-05-19-m2-adr-001-backend-selection/reviewer-report.md`.

## Required Fixes

Add the structural test; reformat ADR-001's decision lines; complete lifecycle reports or adjust the lifecycle oracle by reviewed contract patch; record explicit human approval before treating ADR-001 as locked; preserve critical-review dissent verbatim; document the GT4Py artifact ownership exception; fix the stale memory evidence reference.

## Unresolved Risks

Nsight profiler permission remains unavailable; GT4Py is excluded by toolchain failure rather than benchmark evidence; real M5 physics may invalidate the analytic column surrogate.

Next decision needed: manager must decide whether to apply the required fixes and resubmit this sprint for reviewer acceptance, or keep ADR-001 as proposed and defer backend lock until explicit human approval is recorded.

Decision: Accept with required fixes
