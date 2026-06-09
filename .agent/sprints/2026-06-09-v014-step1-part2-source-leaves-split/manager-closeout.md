# Manager Closeout

## Outcome

Closed as a validated proof/localization sprint.

Merge Decision: accept and commit proof artifacts only. No production source
changes were made.

Verdict:
`STEP1_PART2_SOURCE_LEAVES_LOCALIZED_UPDATE_PHY_TEN_RAW_RTH_TO_T_TENDF_MISSING_IN_JAX_DRY_BUNDLE`.

The previous Step-1 boundary is now split more narrowly: WRF creates the
material `T_TENDF` difference inside `first_rk_step_part2` by adding active raw
`RTH*TEN` leaves in `update_phy_ten`. The current JAX dry source bundle does not
carry equivalent raw leaves. The active leaves for this case are `RTHRATEN` and
`RTHBLTEN`; `RTHBLTEN` is the dominant active raw leaf.

## Evidence

- `proofs/v014/step1_part2_source_leaves_split.py`
- `proofs/v014/step1_part2_source_leaves_split.json`
- `proofs/v014/step1_part2_source_leaves_split.md`
- `proofs/v014/step1_part2_source_leaves_split_wrf_patch.diff`
- `.agent/reviews/2026-06-09-v014-step1-part2-source-leaves-split.md`

Key numbers:

- WRF `update_phy_ten`: `T_TENDF == pre + active RTH`, nested-interior max_abs
  `0.0`.
- WRF `conv_t_tendf_to_moist`: nested-interior max_abs
  `0.00016236981809925055`.
- post-conversion equals `after_first_rk_step_part2`: nested-interior max_abs
  `0.0`.
- current patched-init JAX dry `T_TENDF`: max_abs `2457.5830078125`, RMSE
  `21.674279301376934`.

## Manager Validation

- `python -m py_compile proofs/v014/step1_part2_source_leaves_split.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_part2_source_leaves_split.py`
- `python -m json.tool proofs/v014/step1_part2_source_leaves_split.json >/tmp/manager_step1_part2_source_leaves_split.validated.json`
- `git diff --check`

All passed. No GPU was used and no production source was edited.

## Next Decision

Open an implementation sprint to add true WRF dry physics source leaves for
active `RTHRATEN`/`RTHBLTEN` before `_augment_large_step_tendencies`. Gate the
fix on the same Step-1 proof moving from the current residual to near-zero, then
rerun the strict short grid-field falsifier before any Switzerland, TOST, FP32
R1/R2, or long validation campaign.

## Risks

- The fixture used direct single-rank CPU because `mpirun` failed on PMIx socket
  setup in the sandbox. WRF still completed successfully and emitted full-domain
  d02 truth.
- Inactive WRF `RTH*TEN` leaves can contain uninitialized values; only
  `active_theta_source_flags` leaves were used for causal ranking.
