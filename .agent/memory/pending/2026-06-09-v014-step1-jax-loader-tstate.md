# V0.14 Step-1 JAX Loader T-State

Date: 2026-06-09 15:44 WEST

The active v0.14 grid-parity sprint is
`.agent/sprints/2026-06-09-v014-step1-jax-loader-tstate`.

Trigger proof: `proofs/v014/step1_pre_part1_handoff.*`, verdict
`STEP1_PRE_PART1_LOCALIZED_JAX_LOADER_T_STATE`.

Known ruled out:

- WRF solve_em pre-call `T_STATE` mutation: max_abs `0.0` from
  `after_step_increment` to `before_first_rk_step_part1_call`.
- WRF `first_rk_step_part1` as the source: prior part1 proof shows no material
  internal `T_STATE` mutation.
- Full-vs-perturbation theta mapping: WRF `grid%t_2` maps to
  `State.theta - 300 K`.

Active hypothesis to split:

- raw d02 `wrfinput` theta;
- live-nest base initialization updating `PB/PHB/MUB` without corresponding
  theta/T-state semantics;
- parent boundary package;
- initial `OperationalCarry`;
- haloed step-entry state.

Do not resume TOST, Switzerland, FP32, memory source work, or GPU validation
until this grid-field divergence stage is explained or fixed.
