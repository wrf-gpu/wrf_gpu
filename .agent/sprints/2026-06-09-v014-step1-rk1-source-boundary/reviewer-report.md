# Reviewer Report

## Decision:

Accept. The sprint used the right wall-clock method: extend the focused Step-1
truth/comparator instead of running another long validation or continuing
acoustic debugging. It produced a narrower, falsifiable boundary.

## Findings

- Verdict is
  `STEP1_RK1_SOURCE_LOCALIZED_FIRST_RK_STEP_PART1_PHYSICS_STATE_MUTATION_T_STATE`.
- First material boundary is `after_first_rk_step_part1`, field `T_STATE`.
- The WRF field mismatches both JAX operational carry and
  `_physics_step_forcing.state`, so this is not closed by choosing either
  existing JAX state surface.
- Residual size is material: max_abs about `5.49` K and RMSE about `1.92` K.
- RK1 `small_step_prep` continuity for `T_WORK` and `P_WORK` remains exact.
- Production `src/gpuwrf/**` remained unchanged.

## Weaknesses

The WRF truth does not yet split the inside of `first_rk_step_part1`, so it
cannot identify the exact Fortran call or leaf responsible for `T_STATE`.

## Required Next Sprint

Instrument or compare the internal surfaces inside WRF `first_rk_step_part1`
against the JAX physics adapter output. The next proof should name the exact
T-state mutation source, not continue with acoustic, TOST, Switzerland, FP32, or
memory work.
