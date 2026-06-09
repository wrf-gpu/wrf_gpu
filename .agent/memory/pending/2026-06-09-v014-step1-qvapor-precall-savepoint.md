# V0.14 Step-1 QVAPOR Pre-Call Savepoint

Opened 2026-06-09.

Purpose: extend the disposable CPU-WRF Step-1 pre-call hook to emit
same-boundary `QVAPOR` at `before_first_rk_step_part1_call`.

This is a truth-generation sprint only. It must not edit `src/gpuwrf/**` and
must not resume TOST/Switzerland/FP32/memory work. If successful, rerun the
live-nest theta semantics proof against same-boundary `T_STATE` and `QVAPOR`
before any production theta patch.
