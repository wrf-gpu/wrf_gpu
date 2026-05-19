# Memory Patch Proposal

## Scope

One new auto-memory entry: project's chosen v0 backend (JAX) + the per-scheme Triton fallback gate. This is durable knowledge future Claude turns (post-context-compaction) will need when working on M3+ sprints — they should know without re-reading ADR-001 that the project is committed to JAX.

## Evidence

- ADR-001 selects JAX (pending user approval).
- 5-way M2 evidence: JAX has lowest register count on both bakeoff problems, zero local memory spill on column, fuses to 1 launch per problem, first-pass agent success, highest occupancy on stencil.
- Per-scheme Triton fallback gated by mini-ADR per `.agent/rules/cross-model-review-policy.md`.
- M5 stop/go gate: `local_memory_bytes ≤ 256, registers ≤ 128, kernel_launches ≤ 10` for the first real physics scheme; failure triggers Triton fallback mini-ADR or escalation to user.

## Proposed Destination

New file: `/home/enric/.claude/projects/-home-enric-src-wrf-gpu2/memory/project_backend_decision.md`. Indexed in `MEMORY.md`.

## Patch

```markdown
---
name: Backend decision (ADR-001): JAX primary v0 with gated Triton fallback
description: Per ADR-001 the v0 GPU-native NWP rewrite uses JAX + XLA as the primary backend; Triton is a per-scheme M5 fallback gated by mini-ADR
type: project
---

ADR-001 (2026-05-19, in `.agent/decisions/ADR-001-backend-selection.md`) selects **JAX + XLA** as the primary v0 backend for the GPU-native NWP rewrite. Pin: `jax[cuda13]==0.10.0`. Status as of 2026-05-19: *proposed, pending explicit user approval at M2 closeout*; M3 implementation cannot start before that approval per the constitution's irreversible-decision rule.

Per-scheme Triton fallback (`triton==3.7.0` + `torch==2.12.0`) is available as a contingency at M5 if XLA register-spills on a real physics scheme (Thompson, MYNN, etc.) and JAX-side restructuring fails. The fallback is **gated**: requires a mini-ADR at `.agent/decisions/ADR-001-FALLBACK-<scheme>.md` with cross-model review per `.agent/rules/cross-model-review-policy.md`. No new full ADR is required — the architectural authorization for the hybrid form is granted by ADR-001 — but the per-scheme decision is not silent.

M5 stop/go gate (binding on M5 closeout): for the first real physics scheme implementation, profile must show `local_memory_bytes ≤ 256` AND `registers_per_thread ≤ 128` AND `kernel_launches ≤ 10` AND correctness pass. Otherwise trigger the Triton fallback mini-ADR. If Triton also fails the thresholds for that scheme → escalate to user as a project-scope decision.

**Why this matters across conversations:** future Claude turns working on M3, M4, M5 sprints should default to JAX kernel implementations (`@jax.jit` + `jax.numpy`) without re-deriving the choice; the contingencies above are the only triggers for reconsidering.

**Out of scope for v0:** multi-GPU, AMD/Intel portability, GT4Py (excluded by toolchain failure on Python 3.13).
```

## Reviewer Status

Reviewer Status: not required — backend selection ratified via ADR-001's own cross-model review chain (Codex critical-review + binding reviewer). This memory entry is a durable echo of ADR-001's decision for post-compaction continuity, not a new claim. Manager applies directly when committing this closeout.
