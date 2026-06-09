# Manager Closeout

Merge Decision: accept and land the proof artifacts.

Objective:

Trace the confirmed h10 d02 pre-RK input-boundary mismatch through the JAX
checkpoint/prestep carry producer and previous-step handoff path without
editing production model source.

Accepted verdict:

`PRODUCER_WRITES_BAD_FINAL_CARRY`.

Accepted evidence:

- `proofs/v014/prestep_carry_source_trace.py`
- `proofs/v014/prestep_carry_source_trace.json`
- `proofs/v014/prestep_carry_source_trace.md`
- `.agent/reviews/2026-06-09-v014-prestep-carry-source-trace.md`

Manager validation:

- Python compilation.
- CPU-only proof rerun.
- JSON validation.
- Scoped status check for the allowed output files.

Key finding:

The step-5999 checkpoint file is not corrupting the target leaves. Serialization
and readback preserve `T/P/PB/MU/MUB` exactly. The bad values are already in the
`OperationalCarry` produced by the live nested replay and passed to
`write_checkpoint(..., runtime_state=d02_carry)`.

Roadmap effect:

The next debug target is the previous-step/producer handoff path. Current-step
RK/acoustic, `small_step_finish`, post-RK refresh, history-source remapping,
TOST, Switzerland, and FP32 source work remain blocked behind this grid-parity
root-cause chain.

Next decision:

Open a previous-step handoff bisection sprint that reproduces the producer path
and snapshots d02 around steps 5997, 5998, and 5999 before/after parent force
and child advance. Source-changing fixes remain premature until that sprint
names the first bad write or handoff.
