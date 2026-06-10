# V0.14 Step-1 Dry Source-Leaf Fix

Verdict: `DRY_SOURCE_LEAF_PLUMBING_ACTIVE_BUT_STEP1_T_TENDF_NOT_CLOSED`.

## Evidence

- Patched source-leaf mode is active (`rad_rk_tendf=1`) and emits nonzero JAX dry `T_TENDF`; max_abs `1150.2022428527164`.
- Primary Step-1 residual after WRF `conv_t_tendf_to_moist` vs patched JAX dry `T_TENDF`: max_abs `1497.6112512148795`, rmse `13.468453371786723`.
- WRF active leaves remain much larger: top leaf `RTHBLTEN` max_abs `2522.90576171875`; JAX source-leaf summary max_abs `1150.2022428527164`.
- Forcing radiation on only moves after-conv residual to max_abs `1494.2972624261442`, so radiation cadence is secondary to `RTHBLTEN` fidelity.
- WRF moist-theta conversion is now represented in source-leaf mode, but it is secondary: WRF `after_update` vs `after_conv` max_abs `224.50967407226562`, rmse `4.572429855170764`.

## Ranked Blockers

- `BLOCKING` rank 1: JAX MYNN `RTHBLTEN` source is not WRF-compatible at this Step-1 boundary.
- `SECONDARY_BLOCKING` rank 2: The held JAX `rthraten` leaf is zero at Step-1 while WRF has active `RTHRATEN`; forcing radiation only improves the residual marginally.
- `IMPLEMENTED_STILL_SECONDARY` rank 3: WRF `conv_t_tendf_to_moist` and its `QV_TEND` term are represented in the JAX dry source bundle, but cannot close while MYNN `RTHBLTEN/RQVBLTEN` are too weak.

## Next Boundary

Next source boundary: split MYNN PBL adapter/kernel inputs and outputs against WRF `RTHBLTEN`/`RQVBLTEN`; held `RTHRATEN` and `conv_t_tendf_to_moist` are ranked secondary by the current proof.

Proof objects: `proofs/v014/step1_dry_source_leaf_fix.json`.
