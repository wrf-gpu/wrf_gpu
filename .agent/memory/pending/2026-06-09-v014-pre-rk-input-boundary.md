# V0.14 Pre-RK Input Boundary

Status: pending memory review.

Lesson:

For the h10 d02 grid-parity divergence, explicit CPU-WRF step-6000 pre-RK
input-boundary truth proves the produced JAX step-5999 prestep carry is already
wrong before current-step physics/RK. Do not debug this first as a current-step
RK/acoustic, `small_step_finish`, post-RK refresh, or history-source remapping
problem.

Evidence:

- `proofs/v014/pre_rk_input_boundary.json`
- Verdict: `PRE_RK_INPUT_JAX_PRESTEP_MISMATCH_CONFIRMED`
- First mismatch: `T` max_abs `6.218735851548047`, RMSE
  `4.638818160588427`
- Other mismatches: `P`, `PB`, `MU`, `MUB`

Next use:

Start the next sprint at the JAX checkpoint/prestep carry producer and the
previous-step state handoff path.
