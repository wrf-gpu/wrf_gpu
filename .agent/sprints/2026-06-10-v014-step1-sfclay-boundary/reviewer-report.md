# Reviewer Report

Reviewer Status: ACCEPT_AS_NARROWING_FIX

Decision: ACCEPT_AS_NARROWING_FIX

Summary: The code change is narrow, WRF-sourced, JIT-compatible, and
proof-backed. It does not close Step-1 parity, but it removes one false frontier
and names the next exact boundary.

## Review Notes

- The new `first_timestep` flag is threaded through d02 replay and operational
  Step-1 physics without host/device transfers.
- Warm-step behavior remains default for existing callers.
- The production change is compatible with static-shape JAX tracing: the branch
  is a scalar boolean/JAX predicate consumed by `jnp.where`.
- Proof artifacts explicitly show improvement and non-closure, avoiding a false
  green claim.

## Accepted Evidence

- WRF source semantics are mirrored for UST, MOL, land QSFC, and z/L seed.
- UST and qv-flux boundary errors improve.
- Strict Step-1 remains red and the blocker is narrowed to TSK/ZNT sourcing.

## Required Next Review

The next sprint must not assume TSK/ZNT sourcing without a WRF hook. It must
emit the tiny surface-driver hook and compare exact incoming/outgoing arrays.
