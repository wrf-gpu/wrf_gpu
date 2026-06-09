# V0.14 Prestep Carry Source Trace

Status: pending memory review.

Lesson:

For the h10 d02 grid-parity divergence, checkpoint serialization is not the
cause. Raw pickle runtime state, checkpoint API runtime state, top-level State
payload, and a local round-trip preserve `T/P/PB/MU/MUB` exactly. The bad values
are already in the live nested replay `OperationalCarry` written by the
producer at completed d02 step 5999.

Evidence:

- `proofs/v014/prestep_carry_source_trace.json`
- Verdict: `PRODUCER_WRITES_BAD_FINAL_CARRY`
- Serialization verdict: `CHECKPOINT_READ_WRITE_PRESERVES_TARGET_LEAVES`
- Field max_abs vs CPU-WRF pre-RK truth:
  - `T`: `6.218735851548047`
  - `P`: `589.6789731315657`
  - `PB`: `1047.015625`
  - `MU`: `267.01919069732367`
  - `MUB`: `1050.3046875`

Next use:

Start from the previous-step producer handoff path, not checkpoint I/O or
current-step RK/acoustic. The next sprint should capture d02 around steps 5997,
5998, and 5999 before/after parent force and child advance.
