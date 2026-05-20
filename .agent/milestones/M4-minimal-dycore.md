# M4 - Minimal Dycore

Goal: prove a reduced dycore path.

Deliverables:

- reduced RK/advection/acoustic path
- analytic tests
- WRF-like fixture checks
- profiler report

Acceptance gates:

- tier 1-3 validation passes for selected cases
- precision choices documented

## Reviewer Decision

Accepted 2026-05-20 by manager under user-delegated overnight autonomy. Single sprint (M4-S1) closed across 2 worker attempts + 3 tester attempts (1 hibernate-stuck, 1 watch-loop-stuck, 1 clean Accept) + 2 reviewer (Reject → Accept-with-required-fixes). 384 tests passing. Zero post-init transfers + zero-byte HLO debug-vs-stripped diff verified independently by tester and reviewer. M5 stop/go gate trips on `kernel_launches_per_step=24` (reporting-only per ADR-001, opens per-scheme Triton consideration if needed). ADR-003 dycore-precision finalized; codex critical-review pending. Three documented residual limits (debug-snapshot host-callback per-stage not JAX-ring; acoustic-substep call-shape proxy; tier-2 mass-evidence is tracer surrogate) carried forward to M5+ per `.agent/decisions/MILESTONE-M4-CLOSEOUT.md`.
