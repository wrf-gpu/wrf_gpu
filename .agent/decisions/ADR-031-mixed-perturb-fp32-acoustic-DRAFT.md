# ADR-031 - Mixed Perturbation FP32 Acoustic Mode

**Status**: DRAFT
**Date**: 2026-06-08
**Decision owner**: Manager
**Worker**: GPT-5.5 xhigh R0/R1 de-risk lane
**Applies to**: v0.14 FP32 acoustic roadmap R0/R1

## Context

The v0.13 production path remains fp64 and is gated by the active TOST release
candidate. The v0.14 roadmap asks whether an opt-in acoustic mode,
`acoustic_precision_mode = "mixed_perturb_fp32"`, can be implemented through a
small sequence of proofable sprints without changing the current fp64 path.

The unsafe failure mode is not "fp32 arithmetic exists"; it is representing
small acoustic perturbations as residuals after subtracting large absolute
totals in fp32. The current code still reconstructs base/reference leaves from
`total - perturbation` in small-step prep/finish, operational acoustic staging,
refresh, legacy coupled-core reconstruction, and carry initialization. The
static audit proof records 25 such source lines.

## Decision

Proceed with a v0.14-only, opt-in mixed acoustic lane, but only in this order:

1. R0 contract:
   - Keep `fp64_default` as the default and production label.
   - Add `mixed_perturb_fp32` only as an explicit static mode.
   - Include the mode in report labels and JIT/static cache identity.
   - Fail closed on unknown mode strings.
   - Do not let any current CLI or production wrapper select mixed mode by
     default.
2. R1 explicit-base plumbing:
   - Thread `BaseState` or equivalent explicit base leaves through
     `small_step_prep_wrf`, `small_step_finish_wrf`, pressure diagnostics,
     acoustic staging, refresh, boundary staging, and restart/init assembly.
   - For mixed mode, forbid base recovery by fp32 total-minus-perturbation inside
     the timestep loop.
   - Preserve the legacy fallback only for `fp64_default` and existing tests until
     the explicit-base path has parity proof.
3. R2+ perturbation-authoritative implementation:
   - Make `p'`, `ph'`, `mu'`, pressure memory, and WRF small-step work arrays
     authoritative in the acoustic loop.
   - Reconstruct absolute totals only at controlled interfaces: output, restart,
     boundary exchange, diagnostics, and non-acoustic physics adapters.

This ADR does not accept a full mixed implementation. It accepts only the R0
static contract scaffold and the R1 plan.

## Default-Inert Prototype

Implemented in this lane:

- `AcousticPrecisionMode` labels in `src/gpuwrf/contracts/precision.py`.
- `OperationalNamelist.acoustic_precision_mode`, defaulting to `fp64_default`.
- The mode rides in `OperationalNamelist.tree_flatten()` static aux, so future
  mixed mode has a separate JIT/cache variant.
- No dynamics module or operational carry consumes the label.

## FP64 Islands Retained Initially

Keep these fp64 until individually demoted with before/after proof:

- base/reference pressure, geopotential, and dry-mass leaves,
- lateral boundary reference/base leaves,
- pressure EOS refresh and `diagnose_pressure_al_alt`,
- `calc_p_rho` bracket and smdiv pressure memory,
- horizontal terrain pressure-gradient accumulation,
- implicit-w coefficient builder and Thomas solve,
- `w` and `ph'` boundary forcing,
- turbulence/PBL/surface fields already locked by the precision matrix,
- restart and wrfout reconstruction.

## Kill Gates

Kill or reshape the lane if any of these fail:

- default `fp64_default` output or production CLI behavior changes,
- mixed mode needs a global fp32 dtype flip,
- mixed mode recovers perturbation/base values by fp32 absolute-total
  subtraction in the timestep loop,
- tolerances are widened after observing results,
- validation relies only on JAX-vs-JAX self-comparison,
- transfer audit finds hidden host/device transfer inside timestep loops,
- terrain rest, boundary/nesting ph forcing, or implicit-w solve cannot be made
  stable without masking/clamping.

## Alternatives

- Global fp32 state: rejected. It demotes fields that are known or policy-locked
  fp64 and preserves the cancellation-prone total/residual formulation.
- Leave precision as `force_fp64` only: safe for v0.13, but does not address the
  v0.14 memory/performance lane.
- Implement mixed acoustic immediately: rejected for this sprint. The source
  audit shows explicit-base plumbing is a prerequisite.

## Consequences

- Current production remains fp64 by default.
- A future mixed implementation must add proof objects before any source path
  consumes `mixed_perturb_fp32`.
- Cache/profiling reports must include the mode label.
- R1 will touch acoustic prep/finish and operational staging code, so it needs
  focused bit-inert tests for `fp64_default` plus cancellation/oracle proofs for
  mixed mode before manager review.

## Proof Objects

- `proofs/v014/fp32_acoustic_static_audit.py`
- `proofs/v014/fp32_acoustic_static_audit.json`
- `tests/test_operational_namelist_cache_key.py`
- Gate command:
  `JAX_PLATFORMS=cpu PYTHONPATH=src pytest -q tests/test_operational_namelist_cache_key.py`

## Review

This is a draft ADR from the R0/R1 de-risk worker. It needs manager review before
any R1 source plumbing or v0.13 pull-in decision.
