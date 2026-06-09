# Pending Memory Patch: V0.14 H10 Pre-Step Carry Produced

Scope:

Project-memory update for v0.14 same-state JAX-vs-WRF debugging after the
missing h10 pre-step `OperationalCarry` checkpoint was produced and used.

Evidence:

- `proofs/v014/jax_h10_prestep_carry_producer.json` proves a CPU-loadable
  checkpoint at completed `d02` step 5999:
  `/mnt/data/wrf_gpu2/v014_h10_prestep_carry/d02_step5999_full_carry.pkl`.
- The checkpoint contains `OperationalCarry`, paired `OperationalNamelist`,
  grid shape `159 x 66 x 44`, and SHA256
  `0896e4a272cbeaa85d1bb969ecae82b047e75a028df45a87ddab4f4572af8dde`.
- `proofs/v014/jax_h10_prestep_carry.json` now runs the pre-halo comparison and
  reports `JAX_MISMATCH_T`; first mismatch is `T` with max_abs
  `3.3545763228707983` and RMSE `1.0296598586362888`.
- `proofs/v014/wrf_same_state_marker_savepoint.json` and
  `proofs/v014/wrf_post_rk_refresh_localization.json` establish the green WRF
  target and that WRF history `T` comes from `grid%th_phy_m_t0`.

Proposed destination:

After independent review and after the T history/source-attribution sprint,
add a concise entry to `.agent/memory/stable/recurring-gotchas.md`:

- When a same-surface JAX-vs-WRF dynamic comparison first fails on `T`, do not
  assume a theta-update bug until JAX history/source candidates have been
  compared against WRF `grid%th_phy_m_t0` and THM-side candidates. The h10
  pre-step carry checkpoint above is the current canonical JAX same-surface
  starting artifact.

Reviewer Status:

Pending. Do not apply to stable memory until the T history/source-attribution
sprint confirms whether `JAX_MISMATCH_T` is a source/cadence mapping issue or a
real numerical theta-update issue.
