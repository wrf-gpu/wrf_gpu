# Reviewer Report

Decision:

Accept as a read-only localization proof.

What the proof establishes:

- The h10 checkpoint identity matches the prior T-attribution proof lineage.
- The proof-local RK mirror agrees with the existing pre-halo helper for theta
  (`max_abs=0.0`), so the mirror is adequate for this attribution level.
- The first reachable mismatch is already present at the input/prestep theta
  boundary: `T_OLD` versus JAX prestep theta has max_abs
  `6.218735851548047`.
- The same early boundary has `MU` context mismatch max_abs
  `267.01919069732367`.

Important limitation:

The available WRF input/reference surface does not emit explicit step-6000
pre-RK `P/PB/MUB`, so the proof should not be used to claim a production
source fix target yet. It narrows the next evidence need: emit or hook the
explicit pre-RK JAX and WRF input boundary for `T/P/PB/MU/MUB`.

Residual risk:

The result is a selected-patch, CPU-only localization. It is strong enough to
stop work on final `small_step_finish` or history-source mapping as first fix
targets, but not strong enough to edit a dycore operator directly.
