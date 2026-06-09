# Worker Report

Summary: Completed the CPU-only proof/localization sprint. A disposable WRF copy
under `/tmp/wrf_gpu2_step1_part2_source_leaves_split_20260609` was instrumented
inside `dyn_em/module_first_rk_step_part2.F`, compiled, and run single-rank
after MPI launcher sockets were blocked in the sandbox. The run emitted WRF
truth after `calculate_phy_tend`, `update_phy_ten`, `conv_t_tendf_to_moist`, the
adjacent after-part1/after-part2 surfaces, and the source-save sparse surface.

Produced artifacts:

- `proofs/v014/step1_part2_source_leaves_split.py`
- `proofs/v014/step1_part2_source_leaves_split.json`
- `proofs/v014/step1_part2_source_leaves_split.md`
- `proofs/v014/step1_part2_source_leaves_split_wrf_patch.diff`
- `.agent/reviews/2026-06-09-v014-step1-part2-source-leaves-split.md`

Verdict:
`STEP1_PART2_SOURCE_LEAVES_LOCALIZED_UPDATE_PHY_TEN_RAW_RTH_TO_T_TENDF_MISSING_IN_JAX_DRY_BUNDLE`.

Key result: WRF `update_phy_ten` closes exactly as `T_TENDF = pre + active RTH`
on the nested interior, while the current JAX dry bundle remains divergent.
Active raw leaves are `RTHRATEN` and `RTHBLTEN`; dominant active raw leaf is
`RTHBLTEN`. No production `src/gpuwrf/**` files were edited.
