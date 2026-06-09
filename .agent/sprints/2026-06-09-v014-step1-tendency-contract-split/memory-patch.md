# Memory Patch

Project memory update for the v0.14 Step-1 tendency contract split.

Reviewer Status: accepted for project-memory update after manager rerun.

## Durable Fact

The remaining Step-1 tendency-family divergence is localized to WRF
`first_rk_step_part2` theta source-leaf construction:

- verdict:
  `STEP1_TENDENCY_CONTRACT_LOCALIZED_FIRST_RK_STEP_PART2_T_TENDF_SOURCE_LEAVES`;
- full-domain material field: `T_TENDF` at WRF `after_first_rk_step_part2`;
- `rad_rk_tendf=1` is falsified as the dominant explanation;
- source-save confirms the residual is before dry boundary/spec/acoustic work.

## Next Boundary

Instrument WRF `first_rk_step_part2` internals after `calculate_phy_tend`,
`update_phy_ten`, and `conv_t_tendf_to_moist`. Include raw `RTH*TEN`,
`T_HIST_SRC`, and the current JAX dry source bundle. No `_augment` or
boundary/spec source edit should be made before this split.

## Files To Update

- `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`
- `PROJECT_PLAN.md`
