# Reviewer Report: V0.14 Step-1 Live-Nest Theta/QV Wiring

## Findings

- The patch is scoped to the allowed production file:
  `src/gpuwrf/integration/d02_replay.py`.
- `build_replay_case(..., live_nest_parent=...)` now resolves `use_theta_m`,
  computes WRF's transient post-`blend_terrain`/pre-`start_domain` `MUB`, and
  applies `_wrf_live_nest_adjust_tempqv` before the child `State` is created.
- Final BaseState semantics are preserved: the transient `MUB` is used only for
  `adjust_tempqv`; post-`start_domain` `PB/PHB/MUB` remain the runtime base.
- The production helper closes the prior theta/QV initialization gap against
  same-boundary WRF pre-call truth.
- The larger Step-1 comparison remains red, so this is not a grid-parity close.

## Correctness Risks

The proof uses CPU-only exact helper execution plus static source wiring checks
because full `build_replay_case` still calls GPU-only `State.zeros` in this
environment. That is acceptable for this sprint because the exact helper
outputs are what production consumes, but a future loader cleanup should make
the full real-case constructor CPU-proofable.

The remaining residual is not hidden: Step-1 first divergent schema field is
`T`, largest residual is `P` max_abs `974.9820434775493`, with `PH/MU/W/U`
material. This blocks TOST, Switzerland validation, FP32 source work, and memory
follow-ups.

## Performance Risks

The source patch is initialization-only and does not add host/device transfer to
the timestep loop. It does not challenge the GPU-resident performance model.

## Required Fixes

No additional fixes are required before committing this sprint. The next sprint
must localize the remaining Step-1 `P/PH/MU` boundary/operator residual and
must account for first divergent `T`.

Decision:

Approve merge of the production initialization fix, regenerated proof artifacts,
and sprint closeout. Keep v0.14 validation paused behind Step-1 grid parity.
