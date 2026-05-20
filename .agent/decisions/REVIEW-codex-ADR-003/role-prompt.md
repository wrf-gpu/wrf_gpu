# Role: critical-review (codex gpt-5.5 xhigh)   Decision: ADR-003 Dycore Precision

## Read order

1. `PROJECT_CONSTITUTION.md`
2. `AGENTS.md`
3. `PRECISION_POLICY.md`
4. `PROJECT_PLAN.md` (especially M4/M5 sections)
5. `.agent/decisions/ADR-001-backend-selection.md` (M5 stop/go gate)
6. `.agent/decisions/ADR-002-state-layout.md` (SoA, fp64 default)
7. `.agent/decisions/ADR-005-first-physics-suite.md` (Thompson microphysics is first M5)
8. `.agent/decisions/MILESTONE-M4-CLOSEOUT.md` (3 documented residual limits)
9. `.agent/decisions/ADR-003-dycore-precision.md` — the decision draft you are reviewing
10. `artifacts/m4/{tier1_advection_parity,tier2_invariants,tier3_convergence,dycore_profile,transfer_audit,spacetime_budget}.json` — fp64 evidence
11. `.agent/sprints/2026-05-19-m4-dycore-rk3-advection-acoustic/{worker-report.md,tester-report.md,reviewer-report.md}` — sprint context
12. `.agent/skills/resolving-cross-model-disagreements/SKILL.md`

## Role-specific instructions

Independent senior review of ADR-003. Write to `.agent/decisions/REVIEW-codex-ADR-003/critical-review.md`:

- **Decision**: Accept | Accept with required fixes | Reject
- **Top three structural concerns** with the dycore precision plan
- **Findings** (numbered, severity-ranked, file:line cited)
- **Dissent** if you would have chosen a different precision plan (e.g. fp32 storage with fp64 accumulation in advection from the start; or fp64 retention with no proposed downcasts)
- **Closing recommendation**

Hard rules:
- Read-only everywhere except `.agent/decisions/REVIEW-codex-ADR-003/critical-review.md`.
- Do NOT commit anything.
- Report ≥1500 bytes, must include `Decision:` token.
- Be honest and adversarial.

When done, type `/exit`.
