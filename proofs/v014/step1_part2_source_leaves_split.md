# V0.14 Step-1 Part2 Source Leaves Split

Verdict: `STEP1_PART2_SOURCE_LEAVES_LOCALIZED_UPDATE_PHY_TEN_RAW_RTH_TO_T_TENDF_MISSING_IN_JAX_DRY_BUNDLE`.

## Evidence

- WRF `update_phy_ten`: `T_TENDF == pre + active RTH` on nested interior, max_abs `0.0`, rmse `0.0`.
- WRF `conv_t_tendf_to_moist`: moist-theta formula closes on nested interior, max_abs `0.00016236981809925055`, rmse `8.089162788029723e-07`.
- `after_conv_t_tendf_to_moist` equals `after_first_rk_step_part2` on nested interior, max_abs `0.0`.
- Current patched-init JAX dry `T_TENDF` stays divergent: nested-interior max_abs `2457.5830078125`, rmse `21.674279301376934`.
- Aggregate JAX physics state-delta candidate is also rejected: nested-interior max_abs `1495.1748897121188`, rmse `13.147199610601637`.
- Source-save sparse `T_TENDF` is also divergent vs current JAX dry: max_abs `1326.432250976562`, rmse `97.71886125389001`.
- Source-save is a later adjacent leaf, not the first boundary: vs `after_first_rk_step_part2` max_abs `1199.2587877810001`.
- Dominant active raw leaf is `RTHBLTEN` with nested-interior max_abs `2522.90576171875`.
- Largest active aggregate is `RTH_ACTIVE_SUM_MOIST` with nested-interior max_abs `2562.844970703125`.

## Ranking

- `SUPPORTED` rank 1: WRF raw active RTH source leaves are missing from the current JAX dry bundle.
- `FALSIFIED` rank 2: moist-theta conversion after update_phy_ten is the first wrong boundary.
- `FALSIFIED` rank 3: boundary, spec-zone, or acoustic code mutates T_TENDF before the accepted final surface.
- `FALSIFIED` rank 4: the aggregate JAX physics state delta can be used as a narrow T_TENDF source fix.

## Next Boundary

Implement true WRF dry physics source leaves for active RTHRATEN/RTHBLTEN before `_augment_large_step_tendencies`; do not use aggregate post-physics state deltas unless a scheme-level raw-leaf proof closes this same gate.

Proof objects: `proofs/v014/step1_part2_source_leaves_split.json` and `proofs/v014/step1_part2_source_leaves_split_wrf_patch.diff`.
