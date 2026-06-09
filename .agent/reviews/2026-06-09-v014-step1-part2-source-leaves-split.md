# Review: V0.14 Step-1 Part2 Source Leaves Split

Finding: Step-1 `T_TENDF` divergence is not boundary/spec/acoustic timing. WRF creates it inside `first_rk_step_part2` by adding active raw `RTH*TEN` leaves in `update_phy_ten`, then applying moist-theta conversion.

- Verdict: `STEP1_PART2_SOURCE_LEAVES_LOCALIZED_UPDATE_PHY_TEN_RAW_RTH_TO_T_TENDF_MISSING_IN_JAX_DRY_BUNDLE`.
- Current JAX dry source residual: max_abs `2457.5830078125`, rmse `21.674279301376934`.
- Dominant active WRF raw leaf: `RTHBLTEN`.
- Next decision: Implement true WRF dry physics source leaves for active RTHRATEN/RTHBLTEN before `_augment_large_step_tendencies`; do not use aggregate post-physics state deltas unless a scheme-level raw-leaf proof closes this same gate.

No production `src/gpuwrf` files were changed.
