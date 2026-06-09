# V0.14 Step-1 Tendency Contract Split

Verdict: `STEP1_TENDENCY_CONTRACT_LOCALIZED_FIRST_RK_STEP_PART2_T_TENDF_SOURCE_LEAVES`.

## Result

- CPU backend: `cpu`; GPU used: `False`.
- Earliest full-domain material field after patched init: `T_TENDF` at WRF `after_first_rk_step_part2`.
- Full-domain `T_TENDF` vs JAX dry source: max_abs `2457.5830078125`, rmse `21.20870100357482`.
- Source-save pre-addtend `T_TENDF` vs JAX dry source: max_abs `1326.432250976562`, rmse `97.71894474134935`.
- Proof-local `rad_rk_tendf=1` `T_TENDF`: max_abs `2457.5830078125`.
- Boundary/spec-only is too late for the first failure: source-save is before `relax_bdy_dry`, `rk_addtend_dry`, `spec_bdy_dry`, `small_step_prep`, and acoustic updates.

## RK1 Tendency Symptoms

- Full WRF `after_rk_addtend_before_small_step_prep` compared to JAX `_augment_large_step_tendencies` is not an exact boundary because WRF has already passed dry boundary/addtend/spec work.
- Largest full-surface residuals vs JAX augment: `PH_TEND` max_abs `794096.1875`, `RW_TEND` max_abs `131390.78717448562`, `PH_TENDF` max_abs `27082.453125`, `T_TEND` max_abs `6637.764233094031`, `T_TENDF` max_abs `2596.305908203125`.
- `PH_TEND/RW_TEND` assembly is structurally later in the current JAX path than `_augment`; compare the source-save patch against the acoustic-stage candidate before making an `_augment` source edit.

## Next Exact Boundary

WRF first_rk_step_part2 internals: emit after calculate_phy_tend, after update_phy_ten, and after conv_t_tendf_to_moist for the theta source leaves feeding T_TENDF. Include the raw RTH*TEN/T_HIST_SRC contributors and the current JAX dry physics source bundle.

The earliest full-domain material field after patched init is T_TENDF at after_first_rk_step_part2. Source-save confirms it is already nonzero before rk_addtend_dry/boundary/spec/acoustic. A source edit in _augment or boundary code would be later than the first failure.

Secondary: After T_TENDF is fixed, add an exact WRF post-rk_tendency/post-relax_bdy_dry/post-rk_addtend_dry/post-spec_bdy_dry split for T_TEND/PH_TEND/RW_TEND, or compare source-save PH/RW against the JAX acoustic-stage assembly rather than _augment alone.

Detailed metrics are in `proofs/v014/step1_tendency_contract_split.json`.
