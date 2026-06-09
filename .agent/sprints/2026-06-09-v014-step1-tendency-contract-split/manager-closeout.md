# Manager Closeout

## Outcome

Closed as a validated proof/localization sprint.

Merge Decision: accept and commit proof artifacts only. No production source
changes were made.

Verdict:
`STEP1_TENDENCY_CONTRACT_LOCALIZED_FIRST_RK_STEP_PART2_T_TENDF_SOURCE_LEAVES`.

The stale RK1 `P_STATE` frontier remains closed under patched-init capture.
The next exact failure is earlier: WRF `first_rk_step_part2` already populates
material `T_TENDF` source leaves that the current JAX dry source bundle does
not match.

## Evidence

- `proofs/v014/step1_tendency_contract_split.py`
- `proofs/v014/step1_tendency_contract_split.json`
- `proofs/v014/step1_tendency_contract_split.md`
- `.agent/reviews/2026-06-09-v014-step1-tendency-contract-split.md`

Key numbers:

- full-domain `T_TENDF` at WRF `after_first_rk_step_part2` versus JAX dry
  source: max_abs `2457.5830078125`, RMSE `21.20870100357482`;
- source-save pre-addtend `T_TENDF`: max_abs `1326.432250976562`, RMSE
  `97.71894474134935`;
- proof-local `rad_rk_tendf=1` did not move the `T_TENDF` residual;
- boundary/spec-only explanations are too late because source-save is before
  `relax_bdy_dry`, `rk_addtend_dry`, `spec_bdy_dry`, `small_step_prep`, and
  acoustic updates.

## Manager Validation

- `python -m py_compile proofs/v014/step1_tendency_contract_split.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_tendency_contract_split.py`
- `python -m json.tool proofs/v014/step1_tendency_contract_split.json`
- `git diff --check`

All passed. No GPU was used and no production source was edited.

## Next Decision

Open a WRF `first_rk_step_part2` internal split sprint. Emit disposable truth
surfaces after `calculate_phy_tend`, after `update_phy_ten`, and after
`conv_t_tendf_to_moist`, including raw `RTH*TEN` / `T_HIST_SRC` contributors and
the current JAX dry source bundle. Do not patch `_augment_large_step_tendencies`
or boundary/spec code before this earlier source-leaf boundary is split.

## Risks

- Source-save PH/RW evidence is patch-only; full-domain conclusions use the
  full WRF surfaces.
- Full WRF `after_rk_addtend_before_small_step_prep` is post
  `relax_bdy_dry`/`rk_addtend_dry`/`spec_bdy_dry`; JAX `_augment` is not an
  exact post-spec boundary.
