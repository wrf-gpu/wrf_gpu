# M2 - Backend Bakeoff

Goal: choose the primary backend with evidence, not preference.

Deliverables:

- representative stencil in candidate backends
- representative physics column kernel in candidate backends
- correctness report
- profiler JSON
- ADR selecting the initial stack

Acceptance gates:

- same fixture and metrics across candidates ✅ (M2-S1..S6)
- reviewer challenges maintainability and agent success rate ✅ (per-candidate manager-closeouts)
- human approval if decision is hard to reverse — **pending at M2 closeout user report**

## Reviewer Decision

Accepted (manager-side) 2026-05-19, **conditional on explicit user approval of ADR-001**. See `.agent/decisions/MILESTONE-M2-CLOSEOUT.md` for the full proof-object inventory and bakeoff results. 5/6 candidates implemented (gt4py excluded by toolchain failure; documented per `.agent/milestones/ROADMAP.md M2`). Selected backend: **JAX** (ADR-001), with per-scheme gated Triton fallback.
