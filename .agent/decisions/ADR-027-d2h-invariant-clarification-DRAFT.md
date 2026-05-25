# ADR-027 — D2H Invariant Clarification (inter-kernel D2H == 0)

**Status:** DRAFT
**Date:** 2026-05-25
**Author:** Manager (Claude Opus 4.7, 1M-context)
**Triggered by:** M6b D2H grep + D2H warmed re-capture verdicts (commits `tester/opus/m6b-d2h-grep` + `tester/opus/m6b-d2h-warmed-recapture`).

## Context

`PROJECT_CONSTITUTION.md` mandates "no host/device transfers in the timestep loop." The M6b RETRY interpretation tested `total D2H == 0` via Nsight, which produced both false positives (XLA per-call argument-staging at the GpuExecutable launch boundary) and false negatives (would miss inter-kernel D2H if buried in XLA bookkeeping noise).

D2H decomposition from warmed Nsight (5-step capture):
- **Pre-kernel D2H**: ~25 transfers per scan iteration — XLA argument-staging at GpuExecutable launch. **NOT a constitutional violation**: occurs before any compute kernel runs; bounded; doesn't affect timestep latency in steady-state.
- **Inter-kernel D2H**: ~4 transfers per scan iteration — **REAL inside-loop transfers** between compute kernels. **IS a constitutional violation**.

## Decision

The constitutional invariant "no H2D/D2H in the timestep loop" is measured as **inter-kernel D2H == 0** in a warmed Nsight capture, not as `total D2H == 0`.

Operational definition:
- Take a warmed Nsight trace (at least 3 warm-up calls before `cudaProfilerStart`).
- Profile window covers ≥ 5 steps of the operational timestep loop.
- Filter D2H events: count only those occurring **between compute-kernel launches** within the profile window.
- Pre-kernel D2H (XLA argument-staging at executable launch boundary) is **excluded** from the invariant.
- H2D inside the timestep loop is **always forbidden** (no equivalent bookkeeping exception).

## Why this is the right reading

- XLA per-call argument-staging is bounded by carry size, not by step count; it does not scale with simulation length. Profiling overhead, not physics overhead.
- Inter-kernel D2H **does** scale with step count; if a single kernel emits 4 D2H, a 1h forecast at 18s timestep = 200 timesteps × 4 = 800 D2H, each with synchronization latency. This is the failure mode the constitutional rule was designed to prevent.
- The audit memo (`tester/opus/m6b-d2h-grep` Part 2 + 3) decomposed D2H by source kernel and recommended this distinction.

## Consequences

- M6b RETRY acceptance gate measures `d2h_inter_kernel == 0`, not `d2h_total == 0`. Tests updated accordingly.
- Pre-kernel D2H growth is a **performance signal** (carry layout / argument-staging cost) but not a constitutional violation. Document in ADR-026 if it crosses a threshold (e.g., ≥100 transfers / step).
- Inter-kernel D2H is hard-zero; any non-zero count is a STOP condition for any operational acceptance sprint.

## Open questions to resolve at PROPOSED promotion

- Is there a documented XLA-level option to suppress argument-staging at the executable boundary entirely (e.g., aliased carry, persistent device buffers)? If yes, ADR-026 should adopt it and pre-kernel D2H would also become 0.
- The threshold above which pre-kernel D2H becomes "performance bug" worth fixing. Suggestion: 4× carry-field count.

## References

- `tester/opus/m6b-d2h-grep/d2h_localization.md` (the 50-of-53 pre-kernel finding)
- `tester/opus/m6b-d2h-warmed-recapture/d2h_warmed_memo.md` (the 25 pre + 20 inter decomposition)
- `PROJECT_CONSTITUTION.md` (the original invariant)
- `PROJECT_PLAN.md §14.5.1` (the operational-mode binding rules)
