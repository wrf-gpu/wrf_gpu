# Reviewer Report

Decision: ACCEPT_WITH_NARROWED_BLOCKER.

The production change is accepted even though the exact file
`src/gpuwrf/coupling/noahmp_surface_hook.py` was not in the literal initial
surface-layer file list. It is the correct fix site: changing
`surface_layer.py` fallback semantics would perturb already accepted
surface-layer parity oracles, while this patch supplies the correct WRF
`phy_prep` dry inputs at the faulty caller.

Review notes:

- The fix mirrors the established grid-backed
  `physics_couplers._surface_column_view` contract.
- Grid-less callers keep a compatibility fallback and have focused regression
  coverage.
- The proof distinguishes three causal states: buggy moist theta, dry `t_air`
  only, and full `phy_prep`; only the full production path reaches WRF-close
  water fluxes.
- The remaining strict residual is not hidden by tolerance widening. It is
  honestly reported as MYNN-EDMF `RTHBLTEN`.

Next review focus: the next sprint must own the MYNN-EDMF kernel files and prove
or bound `RTHBLTEN` with WRF-anchored d02 evidence before any long validation.
