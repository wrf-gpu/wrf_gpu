# Memory Patch

Scope:

Project-memory update for v0.14 same-state grid-parity debugging after the
prestep carry source trace.

Evidence:

- `proofs/v014/prestep_carry_source_trace.json` reports
  `PRODUCER_WRITES_BAD_FINAL_CARRY`.
- Serialization/load/readback for `T/P/PB/MU/MUB` is exact.
- The produced live nested replay `OperationalCarry` at completed d02 step 5999
  already differs from CPU-WRF step-6000 pre-RK truth.
- Field max_abs values vs CPU-WRF pre-RK truth remain large:
  `T=6.218735851548047`, `P=589.6789731315657`, `PB=1047.015625`,
  `MU=267.01919069732367`, `MUB=1050.3046875`.

Proposed destination:

Create `.agent/memory/pending/2026-06-09-v014-prestep-carry-source-trace.md`.
After the previous-step handoff bisection names the first wrong producer
handoff, condense the durable rule into stable memory if it generalizes beyond
this h10 debug chain.

Reviewer Status:

Pending. Do not apply to stable memory until the next sprint distinguishes
previous-step final carry assembly, parent/child force packaging, and earlier
integration.
