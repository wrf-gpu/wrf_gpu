# Memory Patch: V0.14 Step-1 QVAPOR Pre-Call Truth Schema

Date: 2026-06-09

Pending until sprint close.

Reason:

- The live-nest theta proof found a likely truth-schema gap: accepted WRF
  pre-call `T_STATE` truth does not include same-boundary `QVAPOR`.
- Existing `QVAPOR` truth artifacts may be post-RK or different-boundary and
  must not be silently substituted.

Expected memory after close:

- Record whether same-boundary WRF pre-call `QVAPOR` truth exists.
- If missing, record the exact minimal WRF savepoint needed to close the next
  theta/`adjust_tempqv` proof.
