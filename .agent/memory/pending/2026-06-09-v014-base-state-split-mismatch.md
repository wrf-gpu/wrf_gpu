# V0.14 Base-State Split Mismatch

Date: 2026-06-09

Status: pending stable-memory review.

Fact:

- `proofs/v014/earlier_source_bisect.json` verdict is
  `BASE_STATE_SPLIT_DEFINITION_MISMATCH`.
- The initial d02 JAX `OperationalCarry` matches native `wrfinput_d02` for
  `PB/MUB`, but not CPU-WRF h0/h1/h10 or h10 pre-RK truth.
- CPU-WRF `PB/MUB` are stable across h0, h1, h10 wrfout and the h10 pre-RK
  hook on the target patch, so replay-time drift is not needed to explain the
  bad h10 base carry.
- Worst target base leaf is `MUB`, max_abs `1050.3046875`; `PB` is also wrong
  with max_abs `1047.015625`.

Operational consequence:

- The next fix target is
  `src/gpuwrf/integration/d02_replay.py::build_replay_case` native child
  base-state split construction.
- Do not debug final child advance, `_operational_force`, checkpoint
  serialization, FP32, TOST, or memory cleanup before this source bug is fixed
  or explicitly bounded.
