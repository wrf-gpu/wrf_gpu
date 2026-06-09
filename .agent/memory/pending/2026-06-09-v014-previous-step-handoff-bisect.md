# V0.14 Previous-Step Handoff Bisection

Date: 2026-06-09

Status: pending stable-memory review.

Fact:

- `proofs/v014/previous_step_handoff_bisect.json` verdict is
  `BAD_BEFORE_FINAL_PARTIAL_SUBCYCLE`.
- The final producer-shaped replay exactly matches the existing bad
  d02 step-5999 checkpoint for `T/P/PB/MU/MUB`, so the reproducer is valid.
- The bad target state is already present at d02 completed step 5997 before
  parent step 2000, `_operational_force`, and child steps 5998-5999.
- At that earliest captured surface, `MUB` max_abs vs CPU-WRF h10 pre-RK truth
  is `1050.3046875` and `PB` max_abs is `1047.015625`.

Operational consequence:

- Do not debug `_operational_force`, final child `_advance_chunk`, checkpoint
  serialization, current-step RK/acoustic, or FP32 first.
- Next debug sprint should bisect earlier than d02 step 5997: native load /
  initial carry, segment drift, base-state split, or an earlier handoff hook.
