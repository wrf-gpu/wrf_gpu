# ADR-027 - D2H Invariant Clarification (inter-kernel D2H == 0)

**Status:** PROPOSED
**Date:** 2026-05-25
**Author:** Manager (Claude Opus 4.7, 1M-context), updated by M6b D2H inside-loop bisection worker and M6b RK1+D2H acceptance worker
**Triggered by:** M6b D2H grep, D2H warmed re-capture, and M6b D2H inside-loop bisection.

## Context

`PROJECT_CONSTITUTION.md` mandates "no host/device transfers in the timestep loop." The M6b RETRY interpretation tested `total D2H == 0` via Nsight, which produced both false positives (XLA per-call argument-staging at the GpuExecutable launch boundary) and false negatives (inside-loop scalar D2H hidden among XLA bookkeeping noise).

Warmed Nsight decomposition showed:

- **Pre-kernel D2H**: bounded XLA executable-boundary bookkeeping before the first compute kernel. This is not a constitutional violation, but remains a performance signal.
- **Inter-kernel D2H**: transfers interleaved with compute kernels. These scale with timestep count and are the constitutional violation.

The M6b inside-loop bisection further localized the warmed residual:

- Disabling boundary application left `d2h_inter_kernel=20` over five steps.
- Reducing acoustic substeps from 2 to 1 left `d2h_inter_kernel=20` over five steps.
- Disabling physics reduced `d2h_inter_kernel` from 20 to 15 over five steps.

Therefore the residual emitters are dynamic operational-mode control flow:

- `src/gpuwrf/runtime/operational_mode.py:353-361`: RK-stage `jax.lax.switch`, 3 x 4 B per timestep.
- `src/gpuwrf/runtime/operational_mode.py:374-380`: radiation-cadence `jax.lax.cond`, 1 x 1 B per timestep.

The M6b RK1+D2H acceptance worker then lifted both localized emitters in
`src/gpuwrf/runtime/operational_mode.py`:

- RK stages are statically sequenced instead of dispatched by a dynamic
  `jax.lax.switch`.
- Radiation cadence is segmented outside the timestep scan so the scan body no
  longer branches on a device-resident cadence predicate.

Measured warmed Nsight evidence after that lift:

- `proof_warmed.nsys-rep` parsed through `scripts/m6b_d2h_warmed_recapture.py`.
- `h2d_total=0`.
- `d2h_inter_kernel=0`.
- `d2h_pre_kernel=25`.
- `d2h_total=25`.

Therefore the constitutional timestep-loop transfer invariant is satisfied for
the lifted operational path, although M6b remains blocked by independent RK1
parity and theta-bound gates.

## Decision

The constitutional invariant "no H2D/D2H in the timestep loop" is measured as **inter-kernel D2H == 0** in a warmed Nsight capture, not as `total D2H == 0`.

Operational definition:

- Take a warmed Nsight trace with at least three warm-up calls before `cudaProfilerStart`.
- Profile window covers at least five operational timestep-loop iterations.
- Filter D2H events: count only those occurring between compute-kernel launches within the profile window.
- Pre-kernel D2H is excluded from the constitutional invariant but recorded.
- H2D inside the timestep loop is always forbidden.

## Implementation Guidance

Dynamic scalar predicates inside `jax.lax.scan` are forbidden on operational acceptance paths unless a warmed Nsight proof shows they do not emit inter-kernel D2H.

Known unsafe patterns:

- Dynamic `lax.switch` over per-step or per-stage indices inside the profiled scan body.
- Dynamic cadence `lax.cond` predicates inside the profiled scan body.

Preferred patterns:

- Static RK stage sequencing when RK order is a static namelist value.
- Static forecast segmentation for cadence events such as radiation.
- Device-resident arithmetic for values that remain array data, without converting predicates into host-visible control signals.

## Pre-kernel D2H Policy

This sprint did not identify a project-approved XLA option that suppresses all executable-boundary D2H bookkeeping while preserving the current JAX operational architecture. Aliasing, donated buffers, and persistent device-resident carries were considered as possible future reductions for boundary traffic, but the accepted M6b lift did not depend on an XLA flag; it removed the timestep-loop emitters by making control flow static.

Pre-kernel D2H is a performance bug when either threshold is crossed:

- `d2h_pre_kernel >= 100` in a warmed five-step operational capture, or
- `d2h_pre_kernel > 4 * resident_carry_field_count` for the profiled operational entry point.

The measured post-lift five-step capture had `d2h_pre_kernel=25`, below the
absolute threshold and documented as executable-boundary bookkeeping. Crossing
either threshold requires an ADR-026 follow-up or a dedicated transfer-audit
sprint. It does not by itself fail the constitutional timestep-loop invariant.

## Consequences

- M6b and later operational acceptance gates measure `d2h_inter_kernel == 0`, not `d2h_total == 0`.
- Any nonzero inter-kernel D2H is a STOP condition for operational acceptance.
- The M6b RK1+D2H acceptance worktree made the localized operational control
  flow static and warmed Nsight confirmed `d2h_inter_kernel == 0`.
- D2H acceptance alone is not sufficient for M6 close; the same acceptance
  sprint recorded RK1 parity and theta-bound blockers.

## References

- `.agent/sprints/2026-05-25-m6b-d2h-grep/d2h_localization.md`
- `.agent/sprints/2026-05-25-m6b-d2h-warmed-recapture/d2h_warmed_memo.md`
- `.agent/sprints/2026-05-25-m6b-d2h-inside-loop-fix/proof_bisection_d2h_emitter.txt`
- `.agent/sprints/2026-05-25-m6b-rk1-d2h-acceptance/proof_d2h_warmed_inter_kernel_zero.json`
- `PROJECT_CONSTITUTION.md`
- `PROJECT_PLAN.md §14.5.1`
