# Sprint Contract

Sprint ID: `2026-05-19-m2-adr-001-backend-selection`
Milestone: M2 — Backend Bakeoff
Sequence: S8 (final M2 sprint; produces ADR-001 — the irreversible architecture decision)
Owner: **manager** (writes the ADR directly; not delegated to codex worker)
Cross-model challenge: codex `gpt-5.5` `xhigh` via `dispatch_role.sh critical-review`
Reviewer (binding judgment on the ADR): codex `gpt-5.5` `high`
Tester: not applicable — this sprint produces a decision document, not code.
Approval status: opened 2026-05-19 by manager after M2-S6 closeout. **GT4Py remediation skipped — 5 candidates is sufficient evidence per manager-autonomy directive.**

## Objective

Write `.agent/decisions/ADR-001-backend-selection.md` selecting **one primary backend** (or a documented hybrid) for the GPU-native NWP rewrite. Decision is **irreversible** per `.agent/rules/architecture-decision-policy.md`; the constitution requires human approval — per the manager-autonomy directive, the manager exercises it with a Codex critical-review as the second-opinion gate and reports the call to the user at M2 closeout, not before.

ADR-001 must integrate the evidence from M2-S2..S6 (cuda_tile, cupy_or_numba, kokkos, jax, triton) and the GT4Py "blocked" finding from M2-S1. Per `M2-DONE.md §D`, the ADR file must:
- be ≥2000 bytes
- include literal tokens: `Decision:`, `Selected backend:`, `Dissent`, `Evidence summary`
- name one of the six candidate families OR explicitly "deferred"
- have a companion Codex cross-model critical review at `.agent/decisions/REVIEW-codex-ADR-001.md`

## Non-Goals

- No new code, no new benchmarks. ADR-001 consumes existing M2 evidence only.
- No commitment beyond v0's RTX 5090 target. The ADR may note multi-vendor concerns but the decision is for v0.
- No commitment to a *specific physics scheme* for M5 (that's the M5-S0 decision-gate sprint).
- No backtracking on M1 fixture schema or any frozen contract.

## File Ownership

Manager may create or edit only these paths:

- `.agent/decisions/ADR-001-backend-selection.md` (new — the ADR itself)
- `.agent/decisions/REVIEW-codex-ADR-001/proposal.md` (new — input to the Codex critical-review)
- `.agent/decisions/REVIEW-codex-ADR-001/critical-review.md` (output of the Codex critical-review)
- `.agent/decisions/REVIEW-codex-ADR-001.md` (the consolidated cross-model review pointer)
- `tests/test_adr_001_structure.py` (new — asserts ADR file exists, ≥2000 bytes, contains required tokens, names one of the six candidates)

Manager may NOT modify governance files, candidate artifacts, sprint contracts, fixtures, or source code in this sprint.

## Inputs (the actual evidence)

Manager must read and weigh:

- `artifacts/m2/scout/toolchain_support_matrix.json` + `toolchain_report.md` (S1 readiness)
- `artifacts/m2/cuda_tile/{stencil,column}_profile.json` + `correctness.json` + `maintainability.md` + `agent_success.json` (S2)
- `artifacts/m2/cupy_or_numba/...` (S3)
- `artifacts/m2/kokkos/...` (S4)
- `artifacts/m2/jax/...` (S5)
- `artifacts/m2/triton/...` (S6; note attempt-2 corrected values)
- All 5 per-candidate sprint folders' `manager-closeout.md` § Lessons
- `PROJECT_PLAN.md §5` (bakeoff candidate definitions)
- `PROJECT_CONSTITUTION.md`, `ARCHITECTURE_PRINCIPLES.md`, `PERFORMANCE_TARGETS.md`, `PRECISION_POLICY.md`
- Project memory `project_target_hardware.md` (Blackwell + toolchain pins + ncu permission limitation)
- `.agent/rules/architecture-decision-policy.md` (the rule that makes this irreversible)
- `.agent/rules/cross-model-review-policy.md` (the format for cross-model challenge)

## Acceptance Criteria

All must hold for closeout.

### ADR-001 content
1. `.agent/decisions/ADR-001-backend-selection.md` exists, ≥2000 bytes.
2. Contains literal tokens: `Decision:`, `Selected backend:`, `Evidence summary`, `Dissent`.
3. `Selected backend:` line names exactly one of: `jax | triton | gt4py | kokkos | cuda_tile | cupy_or_numba | hybrid:<spec> | deferred`.
4. `Evidence summary` includes a tabular comparison of all 5 candidates (gt4py blocked, fairly noted) across at minimum: stencil regs/local/occ/wall, column regs/local/occ/wall, kernel_launches, agent_success summary, maintainability one-liner.
5. `Dissent` section preserves the Codex critical-review's disagreements verbatim (or explicitly notes "none").
6. Decision rationale explicitly addresses: (a) the user's pro-JAX intuition, (b) the previous wrf_gpu OpenACC 5.5× ceiling that this rewrite must beat, (c) the M5-physics column-spilling risk that the analytic surrogate does not exercise.

### Codex critical-review
7. `.agent/decisions/REVIEW-codex-ADR-001/proposal.md` exists — the manager's draft ADR as input to Codex.
8. `.agent/decisions/REVIEW-codex-ADR-001/critical-review.md` exists — Codex's findings. Must contain `Decision:` and a `Findings` block.
9. `.agent/decisions/REVIEW-codex-ADR-001.md` exists — a thin pointer file summarizing what was reviewed and what the manager did with each finding.

### Manager response to critical-review
10. Manager updates the ADR to address every Codex `blocker` and `major` finding, or records dissent in the ADR's `Dissent` section with rationale.

### Tests
11. `tests/test_adr_001_structure.py` asserts: ADR file exists, ≥2000 bytes, contains all 4 required tokens, `Selected backend:` matches the regex `^Selected backend: (jax|triton|gt4py|kokkos|cuda_tile|cupy_or_numba|hybrid:.+|deferred)$`.
12. `pytest -q` passes overall.

### Oracles
13. `python scripts/check_m2_done.py` reports candidates_satisfied=5/6 + ADR-001 row present + REVIEW-codex-ADR-001.md present. (M2 oracle will still be `ok: false` because milestone closeout follows in the next manager turn.)
14. `python scripts/check_m1_done.py` still ok.
15. `python scripts/validate_agentos.py` ok.

## Validation Commands

```bash
python scripts/check_m1_done.py
python scripts/check_m2_done.py
python -m json.tool .agent/decisions/REVIEW-codex-ADR-001/critical-review.md 2>/dev/null || head -50 .agent/decisions/REVIEW-codex-ADR-001/critical-review.md
wc -l .agent/decisions/ADR-001-backend-selection.md
grep -E '^Decision:|^Selected backend:|^## Evidence|^## Dissent' .agent/decisions/ADR-001-backend-selection.md
pytest -q tests/test_adr_001_structure.py
pytest -q
git ls-files -z | xargs -0 stat -c '%s %n' | sort -nr | head -5
```

## Performance Metrics

Not applicable — this sprint produces a decision document.

## Proof Object

- `ADR-001-backend-selection.md` + the two REVIEW-codex-ADR-001 files + the structural test.
- Standard lifecycle reports: worker-report (manager-self-report), reviewer-report, manager-closeout, memory-patch. No tester report (n/a for a decision sprint; AC#12 pytest covers this).

## Risks

- **Premature lock-in.** The decision is irreversible. Mitigation: explicit "M5 fallback to <other-candidate>" clause if the chosen primary fails on real physics. Codex critical-review specifically asked to challenge premature lock-in.
- **Codex critical-review may produce strong dissent.** Manager must preserve it verbatim in the ADR's `Dissent` section — that's the whole point of the cross-model gate.
- **Wall-time numbers are noise** at M1 fixture sizes. ADR-001 should NOT lean heavily on `wall_time_s` and should make the reasoning explicit (registers + local_memory + occupancy + agent_success + maintainability are the load-bearing inputs).
- **GT4Py exclusion as "blocked, not disproven"** — ADR-001 must note that a Python 3.12 venv remediation could resurrect gt4py if M5 reveals JAX/Triton inadequate on real physics. Not a reason to delay ADR-001.

## Handoff Requirements

- Manager writes ADR + spawns Codex critical-review via `bash scripts/dispatch_role.sh critical-review .agent/decisions/REVIEW-codex-ADR-001/ --reasoning xhigh`.
- After critical-review returns, manager applies findings or records dissent, dispatches normal codex reviewer for binding judgment on the ADR (using a dispatch_role.sh reviewer call against this sprint folder).
- After reviewer Accept, manager writes manager-closeout + memory-patch, merges into main, pushes, then opens M2 milestone closeout per `.agent/goals/M2-MANAGER-RUNBOOK.md §D`.

## Worker self-report stub

This sprint's "worker" is the manager. Manager writes a minimal `worker-report.md` documenting which evidence was consumed and the chain of reasoning, to satisfy the lifecycle template + close_sprint.py size/token checks.
